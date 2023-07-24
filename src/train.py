# Python built-in modules
import argparse
import json
import math
import os
import pickle

# PyTorch modules
import torch
import torch.nn as nn

# PyTorch add-on modules
import numpy as np

# Local modules
from utils import coco_eval, to_var
from data_loaders import CocoTrainLoader
from models import Encoder2Decoder
from build_vocab import Vocabulary

# PyTorch modules
from torch.autograd import Variable 
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

import torch.distributed as dist
    
def main(args):
    # Fix args.seed to reproduce training results
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
        
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Create results directory
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    # Image preprocessing, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary binary file
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    num_epochs_no_imp = 0 # Number of epochs without improvement
    spice_scores = [] # SPICE per-epoch scores

    # Value trackers from COCO
    best_spice = 0.
    best_rank_vg_probs = {} # Visual grounding probs from COCO's eval
    best_epoch = 1

    # Initialize evaulation data, if previous is present. This was useful in our first runs on Colaboratory that would stop the computation after 12 hours straight of GPU work to recover the previous evaluationd data and carry on from there keeping trace of the best SPICE score up to that point.
    if (args.exec_eval_data_path != ''):
        eval_data_path = args.exec_eval_data_path
        if os.path.exists(eval_data_path):
            with open(eval_data_path, 'rb') as f:
                eval_data = pickle.load(f)

                if eval_data['num_epochs_no_imp'] > 0:
                    num_epochs_no_imp = eval_data['num_epochs_no_imp']

                if len(eval_data['spice_scores']) > 0:
                    spice_scores = eval_data['spice_scores']

                    # Get highest SPICE score
                    spice_id = np.argmax(spice_scores)

                    # Initialize trackers
                    best_spice = spice_scores[spice_id]
                    best_epoch = spice_id + 1 # Assuming spice_scores contains all the information from epoch 0
    
    # Build COCO training data loader with vocabulary
    data_loader = CocoTrainLoader(args.image_dir, args.caption_path, vocab, 
                                  transform, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers) 

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args.embed_size, len(vocab),
                               args.hidden_size) # Embedding size  := d / 2 = 256 (see ArgumentParser below)
                                                 # Vocabulary size := j (~9567)
                                                 # Hidden size     := d = 512
    
    if args.pretrained:
        # If the pre-trained Enc2Dec model is available, then load its data weights
        adaptive.load_state_dict(torch.load(args.pretrained))

        # Get starting epoch. Note that each saved model is named as 'adaptive-<epoch>.pkl', hence we can extract the starting epoch by simply parsing the Enc2Dec model's filename using the below regular expression. If successful, `start_epoch` will be initialized to that value + 1.
        start_epoch = int(args.pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1   
    else:
        start_epoch = 1
    
    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_subs = list(adaptive.encoder.resnet_conv.children())[args.fine_tune_start_layer:]

    # Build a single list with all the parameters from the sub-modules list of the higher N layers of the pre-trained CNN 
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [i for sublist in cnn_params for i in sublist]
    
    # Note: From the documentation:
    """To construct an Optimizer object you have to give it an iterable
    containing the parameters (all of type Variable) to optimize."""

    # Build an Adam optimizer that will be used to fine-tune only those CNN parameters
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.learning_rate_cnn, 
                                     betas=(args.alpha, args.beta))
    
    # Enc2Dec parameters optimization
    #   - Encoder final weigths (to get V and v_g)
    #   - Decoder parameters
    params = list(adaptive.encoder.affine_a.parameters()) \
                + list(adaptive.encoder.affine_b.parameters()) \
                + list(adaptive.decoder.parameters())

    # Decaying learning rate    
    learning_rate = args.learning_rate
    
    # Language model (LSTM) Loss
    # Note: Refer to the paper for training details
    lm_loss = nn.CrossEntropyLoss()
    
    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        lm_loss.cuda()
    
    # Number of batches (display purposes)
    total_step = len(data_loader)
    
    # Start training for a number of epochs equal to := args.num_epochs + 1 - start_epoch
    for epoch in range(start_epoch, args.num_epochs + 1):
        # Decay the learning rate if args.lr_decay epochs reached
        if epoch > args.lr_decay:
            # Decay the learning rate by args.learning_rate_decay_every
            frac = float(epoch - args.lr_decay)/args.learning_rate_decay_every
            decay_factor = math.pow(0.5, frac) # Refer to the paper for this

            # Decay the learning rate
            learning_rate = args.learning_rate * decay_factor
        
        print('Learning rate for Epoch %d: %.6f' % (epoch, learning_rate))

        # Build an Adam optimizer that will be used to fine-tune all the parameters of the Enc2Dec model, except for those from the higher layers of the CNN encoder that are fully dependent on the other previously-defined Adam optimizer.
        optimizer = torch.optim.Adam(params, lr=learning_rate,
                                     betas=(args.alpha, args.beta))

        # Language model training
        print ('------------------Training cycle for Epoch %d----------------' % (epoch))
        for i, (images, captions, lengths, _, _) in enumerate(data_loader): # For every batch in COCO
            # Note: From the documentation:
            """A PyTorch Variable is a wrapper around a PyTorch Tensor, 
            and represents a node in a computational graph. 
            If X is a Variable then X.data is a Tensor giving its value, and X.grad is another Variable holding
            the gradient of X with respect to some scalar value."""
            images   = to_var(images)
            captions = to_var(captions)

            lengths = [cap_len - 1  for cap_len in lengths] # Reduce captions' lengths by 1 to discard <start> token
            targets = pack_padded_sequence(captions[:, 1:], lengths,
                                           batch_first=True)[0] # Removes <pad> from sequences

            # Forward propagate the LSTM
            adaptive.train()

            # Note: From the documentation:
            """In PyTorch, we need to set the gradients to zero 
            before starting to do backpropragation because PyTorch 
            accumulates the gradients on subsequent backward passes. 
            This is convenient while training RNNs."""
            adaptive.zero_grad()

            packed_scores = adaptive(images, captions, lengths) # Same as adaptive.forward(...)

            # Backpropagate the LSTM w.r.t. the cross-entropy loss
            loss = lm_loss(packed_scores[0], targets)
            loss.backward()
            
            # Clip gradients to avoid exploding gradients' problem in RNNs
            for p in adaptive.decoder.LSTM.parameters():
                p.data.clamp_(-args.clip, args.clip)
            
            # Let Adam do an optimization step
            optimizer.step()
            
            # Fine-tune the CNN if args.cnn_epoch has been reached
            if epoch > args.cnn_epoch:
                cnn_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Cross-entropy loss: %.4f, Perplexity: %5.4f' %
                                            (epoch, args.num_epochs, i,
                                             total_step, loss.data.item(),
                                             np.exp(loss.data.item())))  
            
        # Save the adaptive attention KwtL model after each epoch
        torch.save(adaptive.state_dict(), os.path.join(args.model_path, 
                   'adaptive-%d.pkl' % (epoch)))        
      
        # Evaluation on COCO validation set   
        metrics_results, rank_vg_probs = coco_eval(adaptive, args, epoch)

        # Extract SPICE metric
        spice = metrics_results['SPICE']
        spice_scores.append(spice)

        with open(os.path.join(args.results_path,
                               'metrics_results-%d.pkl' % (epoch)), 'wb') as f:
            pickle.dump(metrics_results, f) # Save evaluation scores on Kwtl for each epoch

        # This KwtL model, as opposed to the autors' which focused on the older CIDEr, is trained against the SPICE metric, that is, it is trained in order to maximize the value of the SPICE metric epoch after epoch and, in turn, select through the best SPICE value which model performs best.
        if spice > best_spice:
            print ('New best model at epoch #: %d with SPICE score %.2f' % (epoch, spice))
            best_rank_vg_probs = rank_vg_probs
            best_spice = spice
            best_epoch = epoch

            # Reset as we got improvement
            num_epochs_no_imp = 0
        else:
            num_epochs_no_imp += 1

        # Save intermediate SPICE data for Colaboratory's issue
        eval_exec_scores = {
            "spice_scores": spice_scores,
            "num_epochs_no_imp" : num_epochs_no_imp
        }
       
        with open(os.path.join(args.results_path, 
                               'eval_exec_data-%d.pkl' % (epoch)), 'wb') as f:
            pickle.dump(eval_exec_scores, f)

        if len(spice_scores) > 5:
            last_6 = spice_scores[-6:]
            last_6_max = max(last_6)
            
            # Check if there is improvement over the last 6 SPICE scores, otherwise do early stopping.
            if last_6_max != best_spice:                
                print ('No improvement with SPICE in the last 6 epochs... Early stopping triggered.')
                print ('Model of best epoch #: %d with SPICE score %.2f' % (best_epoch, best_spice))
                break    
    
    # Load the weights of the best model
    adaptive.load_state_dict(torch.load(os.path.join(args.model_path,
                                        'adaptive-%d.pkl' % (best_epoch))))

    # Save the best KwtL model in .pt format (not only the weights) for later use in Android (# TODO)                                        
    torch.save(adaptive, os.path.join(args.model_path,
                                      'adaptive-%d.pt' % (best_epoch)))

    eval_scores = {
        start_epoch: spice_scores
    }

    # Store final validation SPICE scores
    with open(os.path.join(args.results_path,
                           'eval_scores.pkl'), 'wb') as f:
        pickle.dump(eval_scores, f)

    # Store final visual grounding probabilities
    with open(os.path.join(args.results_path,
                           'rank_vg_probs.pkl'), 'wb') as f:
        pickle.dump(best_rank_vg_probs, f)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and general parameters
    parser.add_argument('-f', default='self', help='To make it runnable in Jupyter')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='Path to save trained models')
    parser.add_argument('--results_path', type=str, default='./results',
                        help='Path to save validation results')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='Size to randomly crop images at')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='Path to vocabulary binary file')
    parser.add_argument('--image_dir', type=str, default='./data/resized' ,
                        help='Directory with resized training images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/karpathy_split_train.json',
                        help="Path to COCO's train annotation file")
    parser.add_argument('--caption_val_path', type=str,
                        default='./data/annotations/karpathy_split_val.json',
                        help="Path to COCO's val annotation file")
    parser.add_argument('--exec_eval_data_path', type=str, default='',
                        help='Path to previous evaluation data binary file')
    parser.add_argument('--log_step', type=int, default=10,
                        help="Step interval to print log info")
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for model reproducibility')
    
    # ---------------------------Hyper-parameters setup------------------------------------
    
    # CNN fine-tuning
    # Note: Fine tune only the last 5 layers of CNN to avoid overfitting
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
                        help='Start fine-tuning CNN after these many epochs')
    
    # Adam optimizer parameters
    parser.add_argument('--alpha', type=float, default=0.8,  # Standard choice from PyTorch
                        help='Alpha value in Adam')
    parser.add_argument('--beta', type=float, default=0.999, # Standard choice from PyTorch
                        help='Beta value in Adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='Learning rate for the whole Enc2Dec model')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-4,
                        help='Learning rate for CNN fine-tuning')
    
    # LSTM hyper-parameters
    # Note: Even though the embedding size is not specified in the paper, we opted to put it to 256, that is, half the LTSM's hidden size, because the LSTM's input x_t is generated by concatenating two vectors of embedding size, and the paper referenced its overall dimension as R^d, that is, the very same dimension they referred to for the hidden size.
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Dimension of word embedding vectors and v_g')
    parser.add_argument('--hidden_size', type=int, default=512, # Taken directly from the paper
                        help='Dimension of LSTM hidden states')
    
    # Training details
    parser.add_argument('--pretrained', type=str, default='', help='Start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Total epochs before stopping')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--eval_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Total workers in parallel for Torch')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='Clip gradients to avoid exploding gradients')
    parser.add_argument('--lr_decay', type=int, default=20,
                        help='Decay learning rate after this epoch')
    parser.add_argument('--learning_rate_decay_every', type=int, default=20,
                        help='Decay learning rate every these many epochs')
    
    # Parse arguments and run
    args = parser.parse_args()
    
    print('------------------------Model training details--------------------------')
    print(args)
    
    # Start training
    main(args)
