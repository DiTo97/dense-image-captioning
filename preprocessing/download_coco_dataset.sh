#!/bin/bash

# Data's folder
DATA=./data

# MS COCO annotations ZIP file
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P $DATA

# MS COCO images ZIP file
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P $DATA
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P $DATA

# Unzip MS COCO annotations
unzip $DATA/captions_train-val2014.zip -d $DATA
rm $DATA/captions_train-val2014.zip 

# Unzip MS COCO images into val2014/train2014 folders,
# despite them not necessarily mapping the namesake datasets,
# as the train/val/test split will be inferred by Karpathy's split
unzip $DATA/train2014.zip -d $DATA
rm $DATA/train2014.zip 
unzip $DATA/val2014.zip -d $DATA
rm $DATA/val2014.zip 