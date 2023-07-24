# Python built-in modules
import glob
import json
import pickle

# PyTorch modules
import torch

from torch.autograd import Variable
from torchvision import transforms

# Python add-on modules
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Local modules
from data_loaders import CocoEvalLoader
from build_vocab import Vocabulary

from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap

def to_var(x, volatile=False):
    """Wrap a Torch tensor into a Variable object and move it to GPU, if available.
    Note: A Variable object allows backpropagation of the gradients."""
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, volatile=volatile)

def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with Matplotlib.
    Note: Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    
    Parameters
    ---------
    images: np.ndarray
        List of images as NumPy arrays
    
    cols: int - [1]
        Number of columns to have in figure.
    
    titles: list - [None]
        List of titles corresponding to each image.
    """
    
    # If titles is provided (not None) assert that it has
    # the same length as images
    assert((titles is None) or (len(images) == len(titles)))
    
    n_images = len(images)
    rows = np.ceil(n_images / float(cols)) # Number of rows

    if titles is None: # Generate dummies, if titles isn't provided
        titles = ['Image at #(%d)' % i for i in range(1, n_images + 1)]
        
    fig = plt.figure(figsize=(16, 16))

    # Plot every image with its title
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, cols, n + 1)

        if image.ndim == 2: # Not an RGB image
            plt.gray()
            
        plt.imshow(image)

        a.axis('off')
        a.set_title(title, fontsize = 100)
        
    # Adjust figure size
    fig.set_size_inches(np.array(fig.get_size_inches())
                        * n_images)

    # Flush all images at once
    plt.tight_layout(pad=0.4, w_pad=0.5,
                     h_pad=1.0)
    plt.show()

def coco_eval(model, args, epoch):
    """Evaluate the KwtL adaptive model on MS COCO validation split
    
    Parameters
    ----------
    model : models.Encoder2Decoder
        KwtL adaptive model after the given epoch
        
    args : dict
        Dictionary of arguments
        (i.e., crop_size, vocab_path, etc...)
        
    epoch : int
        Current training epoch

    Returns
    -------
    metrics_results : dict
        Dictionary of per-metric scores
        (i.e., CIDEr, BLEU, SPICE, etc...)

    vg_probabilites: dict
        Visual grounding probabilities for every word in the vocabulary given their frequency within the generated captions for every validation image in COCO's split. Refer to the paper to check how they are calculated starting from the LSTM's sentil gate $Beta$.
    """
    
    model.eval() # Set the model to evaluation mode to avoid
                 # the backpropagation of gradients
    
    # Sequence of transforms to apply to images:
    #   - Crop to 224x224 (args.crop_size)
    #   - Return as a Torch tensor
    #   - Normalize RGB channels using mean and std from ImageNet. This is a standard practice utilized within PyTorch environment, since ImageNet comprises of millions of images and computing the mean and the standard deviation among all those images is much more reliable than doing it from scratch.
    transform = transforms.Compose([ 
        transforms.Scale((args.crop_size, args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load the vocabulary (binary file of a Vocabulary object)
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # COCO validation data loader.
    # By passing a Transform object, the same transformations are applied on all the images contained in args.image_dir. This is done before the actual evaluation/training on the captions contained in args.caption_val_path even takes place.
    eval_data_loader = torch.utils.data.DataLoader( 
        CocoEvalLoader(args.image_dir, args.caption_val_path, transform), 
        batch_size = args.eval_size,
        shuffle = False, num_workers = args.num_workers,
        drop_last = False
    )  

    # Average visual grounding probabilities for every word in the vocabulary
    # w.r.t. the number of generated captions it is present in
    vg_probabilities = {}
    
    # List of generated captions to be compared with ground-truth
    # (i.e., {'image_id': <image_id>, 'caption': <caption>})
    results = []

    print('---------------------Start evaluation on MS-COCO dataset-----------------------')
    for i, (images, image_ids, _ ) in enumerate(eval_data_loader):
        images = to_var(images) # Wrap images in a Variable.
        generated_captions, _, Betas = model.sampler(images)
        
        # If CUDA is available, the below tensors are in GPU. If that's the case, move them to CPU (with `.cpu()`) before transforming them to NumPy arrays, because we don't want to overcrowd the GPUs with the following computations as they are not too demanding.
        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
            Betas = Betas.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()
            Betas = Betas.data.numpy()

        # Build captions based on vocabulary since captions contains the indexes of words from the vocabulary, but not the words themselves. So, to have something readable, they have to be converted using the vocabulary's mapping, while skipping the <end> token.
        for image_idx in range(captions.shape[0]): # For every image in the batch
            sampled_ids = captions[image_idx]
            sampled_caption = []

            betas = Betas[image_idx]
            
            for j, word_id in enumerate(sampled_ids): # For every word index in the caption
                word = vocab.idx2word[word_id]

                if word == '<end>': # Skip the <end> token and break
                    break
                else:
                    sampled_caption.append(word)

                    if word not in vg_probabilities:
                        vg_probabilities[word] = [1, 1. - betas[j]] # k: word, v: [N. of occurences, sum of VG probabilities for that word]
                                                                    # Visual grounding prob := 1 - $Beta$
                    else:
                        vg_probabilities[word][0] += 1
                        vg_probabilities[word][1] += (1. - betas[j])
                    
            sentence = ' '.join( sampled_caption )
            
            # Note: image_idx --> Image index within the current batch
            #       image_id  --> Unique image ID within COCO's dataset
            temp = {
                'image_id': int(image_ids[image_idx]),
                'caption': sentence
            }
            results.append(temp)
        
        # Display evaluation process every 10 batches
        if (i + 1) % 10 == 0:
            print('[%d/%d]'%((i + 1), len(eval_data_loader )))
            
    vg_probabilities = {k: (v[1] / v[0]) for k, v in vg_probabilities.items()} # Compute occurence-based average VG --> k: word, v: avg VG
    vg_probabilities = sorted(vg_probabilities.items(), key = lambda x: x[1], reverse=True) # reverse=True -> Descending order
            
    print('------------------------Captions generated------------------------------------')
            
    # Note: `coco.loadRes()` from COCO's API requires a file path as input to load the generated captions. That's why we need to first save the resulting list of captions to a JSON file and only then load it from the COCO object. This of course slows the evaluation down a bit (as any I/O operation), but there's no way around it.

    # Evaluate the results based on COCO's API
    ann_file = args.caption_val_path
    res_file = 'results/mixed-' + str(epoch) + '.json'
    json.dump(results, open(res_file , 'w'))
    
    coco = COCO(ann_file)
    coco_res = coco.loadRes(res_file)
    
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds() # Filter out images we haven't generated a caption for
    coco_eval.evaluate()
    
    # Get scores for validation evaluation with a focus on SPICE
    metrics_results = {}

    print ('-----------Evaluation performance on MS-COCO validation dataset for Epoch %d----------' % (epoch))
    for metric, score in coco_eval.eval.items():
        metrics_results[metric] = score
        print ('%s: %.4f'%(metric, score))
            
    return metrics_results, vg_probabilities
