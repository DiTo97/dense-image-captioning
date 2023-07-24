import pickle
import argparse

from collections import Counter

from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer # Tokenizer implementation that conforms to the Penn Treebank's tokenization conventions. This tokenizer is a Java implementation of Professor Chris Manning's Flex tokenizer, pgtt-treebank.l. It reads raw text and outputs tokens as edu.stanford.nlp.trees.Words in the Penn treebank format, though over time the tokenizer has added quite a few options and a fair amount of Unicode compatibility.

class Vocabulary(object):
    """Wrapper of COCO's dataset vocabulary that exposes both the word to index and the index to word mappings.
    Note: It will be extended in the future to allow for a generic non-COCO dataset."""

    def __init__(self):
        self.word2idx = {} # Word to index dict
        self.idx2word = {} # Index to word dict

        self.idx = 0 # Unique incremental index

    def add_word(self, word):
        """Assign a word to the corresponding index and viceversa, if it is not defined in the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word

            # Increment the index
            self.idx += 1 

    # The __call__ method enables Python programmers to write classes where the instances behave like functions and can be called like a function. 
    # If this method is defined, x(arg1, arg2, ...) is a shorthand for x.__call__(arg1, arg2, ...).
    def __call__(self, word):
        """Return the index corresponding to the word in the vocabulary. In case that is not defined, then return the index to the <unk> token, which is used for unknown words."""
        if not word in self.word2idx:
            return self.word2idx['<unk>']

        return self.word2idx[word]

    def __len__(self):
        """Return the length of the vocabulary"""
        return self.idx

def build_vocab(json, threshold, batch_size = 8):
    """Build a vocabulary wrapper from a given JSON file compliant with Karpathy's format. In this case it will simply use the training set originated from Karpathy's split as the source of words for the vocabulary, but it could be extended to different formats (i.e., Flickr30k)

    Parameters
    ----------
    json : str
        Path to a Karpathy-like JSON file

    threshold: int
        Minimum number of occurances of a word to be kept in the vocabulary.
        Note: Original authors chose 5 to get around 9750 different words.

    batch_size: int - [8]
        How many images to feed the (PTB) tokenizer with. Remember that each image has 5 captions associated to it as per COCO's standard, so with a batch of 8 images 40 captions at a time will go through the lemmatization process to greatly speed PTB up without making it stall.
    """
    
    coco = COCO(json) # COCO's API wrapper

    # A Counter object can be used to easily count occurrences. It fills and updates a dictionary with (k: word, v: occurences) for every new list of words passed to it. Its usage will help us to spot under-threshold words and to remove them from the dictionary.
    counter = Counter()
    tokenizer = PTBTokenizer()

    img_ids = coco.getImgIds()

    for i in range(0, len(img_ids), batch_size): # Lemmatize batches of COCO's annotations
        sub_img_ids = img_ids[i:i + batch_size]
        ann_ids = coco.getAnnIds(imgIds = sub_img_ids) # Concatenate annotations for all the COCO's images in the batch, since we are not interested here in which specific image they do refer to, but only in the words they are made of to construct a uniform vocabulary.

        captions_to_tokenize = {
            0: [] # 0 as a dummy image ID
        }

        # Build a single input to PTB tokenizer
        for ann_id in ann_ids:
            caption = str(coco.anns[ann_id]['caption'])
            captions_to_tokenize[0].append({
                'caption': caption
            })

        tokens_per_caption = tokenizer.tokenize(captions_to_tokenize)[0] # List of space-separated strings of PTB tokens
        tokens = ' '.join(tokens_per_caption).split(' ')
        counter.update(tokens) # Updates the counts for each word

        if i % 1000 == 0:
            print("[%d/%d] Fully tokenized images." %(i, len(img_ids)))

    # If the word occurs less than 'threshold' times, then filter it out.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper
    vocab = Vocabulary()

    # Add some special tokens.
    # Note: Remember the below indexes to better understand models.py
    vocab.add_word('<pad>')   # 0 - Padding token to make equal-length sequences
    vocab.add_word('<start>') # 1 - Starting token to open a sequence with
    vocab.add_word('<end>')   # 2 - Ending token every sequence must terminate with
    vocab.add_word('<unk>')   # 3 - Unknown word token

    # Add words to the vocabulary.
    for word in words:
        vocab.add_word(word)

    return vocab

def main(args):
    """Build a Vocabulary object by using the specified path for the captions and a threshold. Both values (a string and a number) are retrieved by using the parameter values contained in args, which are either default values or user input."""
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path

    # Saves the Vocabulary object in the specified path as a binary file.
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: %d" % len(vocab))
    print("Saved to '%s'" % vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--caption_path', type=str, 
                        default='./data/annotations/karpathy_split_train.json', 
                        help="Path to COCO's annotation file")
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='Path to save pickled vocabulary to')

    # Vocabulary hyperparameters
    parser.add_argument('--threshold', type=int, default=5, 
                        help='Minimum word count threshold')

    # Parse arguments and run
    args = parser.parse_args()
    main(args)
