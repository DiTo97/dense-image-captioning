#!/bin/bash

# COCO's API folder
COCO=coco
mkdir $COCO

# Clone eval-metrics-rich COCO API
git clone https://github.com/ruotianluo/coco-caption.git 

# Move relevant files to COCO folder
mv coco-caption/pycocotools ./$COCO
mv coco-caption/pycocoevalcap ./$COCO
mv coco-caption/get_google_word2vec_model.sh ./$COCO
mv coco-caption/get_stanford_models.sh ./$COCO

# Delete unnecessary ones
rm -rf coco-caption

cd $COCO

# Downlowad Standford models for SPICE
source get_stanford_models.sh
rm get_stanford_models.sh

# Download Google's models for WMD
source get_google_word2vec_model.sh
rm get_google_word2vec_model.sh

cd ..
