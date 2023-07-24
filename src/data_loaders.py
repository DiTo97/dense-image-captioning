# PyTorch modules
import torch
import torch.utils.data as data

from torchvision import transforms, datasets

# Python built-in modules
import os
import pickle
import string
import json

# Python add-on modules
import numpy as np
import skimage.transform 

from PIL import Image

# Local modules
from build_vocab import Vocabulary
from coco.pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    """COCO custom Dataset object compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None):
        """
        Parameters
        ----------
        root : str
            Path to root images directory

        json : str
            Path to COCO's annotation file

        vocab : build_vocab.Vocabulary
            Vocabulary object

        transform : transforms.Compose
            Compose object of transforms to apply to images
        """

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys()) # COCO annotations IDs
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """
        Return the item from COCO's dataset at that index

        Returns
        -------
        (image, caption, image_id, filename)
            Data item tuple of interest
        """

        coco   = self.coco
        vocab  = self.vocab
        ann_id = self.ids[index]

        # Extract ground-truth caption
        caption  = coco.anns[ann_id]['caption']
        image_id = coco.anns[ann_id]['image_id']

        # Extract filename to load image 
        filename = coco.loadImgs(image_id)[0]['file_name']

        # Decide whether to draw from val2014 or train2014 folder
        if 'val' in filename.lower():
            path = 'val2014/' + filename
        else:
            path = 'train2014/' + filename

        # Load image as RGB and apply transforms, if any
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Convert the ground-truth caption (str) to the corresponding list of word indexes from the vocabulary.
        # FIXME: PTB tokenixer to tokenize ground-truth captions.
        tokens  = str(caption).lower().translate(string.punctuation) \
                              .strip().split()

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption) 

        return image, caption, image_id, filename

    def __len__(self):
        """Return the total number of annotations"""
        return len(self.ids)

class CocoTrainLoader(data.DataLoader):
    """Return torch.utils.data.DataLoader for custom COCO's Dataset object."""

    def __init__(self, root, json, vocab, transform, batch_size,
                       shuffle, num_workers):
        """Setup the DataLoader object's hyperparameters and assign a custom collate function to it which will perform some operations on the items returned by the call to `__getitem__` of the underlying Dataset object, before grouping them in batches.
        
        Parameters
        ----------
        root, json, vocab, transform
            See CocoDataset class docstring

        batch_size : int
            Size of the batches

        shuffle : bool
            Whether to shuffle data before grouping it in batches

        num_workers : int
            Number of parallel workers
        """

        # COCO caption Dataset object
        coco = CocoDataset(root=root, json=json, vocab=vocab,
                           transform=transform)

        def custom_collate_fn(data):
            """Create mini-batch Torch tensors from the list of tuples (image, caption, image_id, filename) returned by CocoDataset.

            We should build a custom collate_fn rather than using the default one, because merging caption (including <pad> tokens) is not supported by default. To address this issue we will compute the length of every reference caption and then take the longest one as a reference to pad all the others, such that they're all of equal length. 

            Parameters
            ----------
            data: list of tuples (image, caption, image_id, filename)
                - image : torch.Tensor
                    Image tensor of shape (3, 256, 256)

                - caption: torch.Tensor
                    Ground-truth caption tensor of variable length.

                - image_id : int
                    Image unique ID within COCO's dataset

                filename : str
                    Image filename

            Returns (batch_size as B)
            -------
            images: torch.Tensor
                Batch of images tensor of shape (B, 3, 256, 256)

            targets: torch.Tensor
                Batch of ground-truth captions of shape (B, padded_length)

            lengths: list
                List (of length B) of the valid length of each padded caption. Basically it keeps track, for every given caption, of how much of its tensor is a true caption and how much is instead the artifically added padding with <pad> tokens, in order to faster retrieve the caption later.

            image_ids: list
                List of unique image IDs from COCO for the batch

            filenames: list
                List of filenames from COCO for the batch
            """

            # Sort data list by caption (x[1]) length
            data.sort(key=lambda x: len(x[1]), reverse=True) # Descending order

            # Unzip separate data items
            images, captions, image_ids, filenames = zip(*data)

            image_ids = list(image_ids)
            filenames = list(filenames)

            # Merge images from 3D tensors to 4D tensors
            images = torch.stack(images, 0)

            # Merge captions from 1D tensors to 2D tensors
            lengths = [len(cap) for cap in captions]
            targets = torch.zeros(len(captions), max(lengths)).long() # targets is a Torch tensor with a number of entries equal to the total number of captions available in COCO's dataset initialized with index 0 (torch.zeros()), that is, the <pad> token for a length equal to the longest caption in COCO.

            for i, cap in enumerate(captions):
                end = lengths[i]
                targets[i, :end] = cap[:end] # Fill every entry with the corresponding caption

            return images, targets, lengths, image_ids, filenames

        super().__init__(dataset=coco, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=custom_collate_fn)
            
class CocoEvalLoader(datasets.ImageFolder):
    """Return a custom DataLoader for evaluation purposes"""

    def __init__(self, root, ann_path, transform=None, 
                 loader=datasets.folder.default_loader):
        """
        Parameters
        ----------
        root : str
            Path to root images directory

        ann_path : str
            Path to COCO's annotation file
        """

        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = json.load(open(ann_path, 'r'))['images']
    
    def __getitem__(self, index):
        """Return the item from COCO's dataset at that index

        Returns
        -------
        (image, image_id, filename)
            Data item tuple of interest
        """

        filename = self.images[index]['file_name']
        image_id = self.images[index]['id']
        
        # Filename for the image
        if 'val' in filename.lower():
            path = os.path.join(self.root, 'val2014' , filename)
        else:
            path = os.path.join(self.root, 'train2014', filename)

        # Load the image
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id, filename

    def __len__(self): 
      return len(self.images)
