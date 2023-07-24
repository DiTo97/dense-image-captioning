"""Perform the standard Karpathy (i.e., NeuroTalk inventor) split on MS COCO captioning data: allocate 5k images for val/test splits respectively, and map all words that occur <= 5 times to a special <unk> token."""

import json

from random import shuffle, seed

seed(123) # Make a reproducible split

NUM_VAL  = 5000
NUM_TEST = 5000

val   = json.load(open('./data/annotations/captions_val2014.json', 'r'))
train = json.load(open('./data/annotations/captions_train2014.json', 'r'))

# 1. Merge together the original splits
imgs   = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

# 2. Shuffle COCO images
shuffle(imgs)

# 3. Build a dict with val, test, train's splits IDs
dataset = {}

dataset['val']   = imgs[:NUM_VAL]
dataset['test']  = imgs[NUM_VAL: NUM_VAL + NUM_TEST]
dataset['train'] = imgs[NUM_VAL + NUM_TEST:]

# 4. Group captions (5 for each image) into lists by image IDs
itoa = {}

for a in annots:
    imgid = a['image_id']

    if not imgid in itoa:
        itoa[imgid] = []

    itoa[imgid].append(a) # Append caption to corresponding
                          # list by image ID

# 5. Fill a JSON file with the respective test/val/train splits'
# information and dump it out in the folder
json_data = {}

info = train['info']
licenses = train['licenses']

splits = ['val', 'test', 'train']

for split in splits:
    json_data[split] = { 
        'type': 'caption',
        'info': info,
        'licenses': licenses,
        'images': [],
        'annotations': []
    }
    
    for img in dataset[split]:
        img_id = img['id']
        anns = itoa[img_id]
        
        json_data[split]['images'].append(img)
        json_data[split]['annotations'].extend(anns) # Generate a single expanded list of annotations where a single image corresponds to 5 different spots of the list. Hence, to gather the image position in its list from a given annotation, you have to divide the annotation's position by 5.
        
    json.dump(json_data[split], open('./data/annotations/karpathy_split_'
                    + split + '.json', 'w')) # Each of these files will equally draw from val2014/train2014 folders for images associated to their namesake dataset. The discriminant between the two folders is dictated by the images' name, as images with 'val' in their name will be searched in the val2014 folder.