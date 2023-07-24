#!/bin/bash

# Clean old Python venv
rm -rf ./CV-env

# Clean COCO's dataset and API
rm -rf ./data
rm -rf ./coco

# Create Python venv on sandbox
apt-get install python3-venv
python3 -m venv CV-env
source CV-env/bin/activate

# Update PIP manager
pip3 install --upgrade pip
pip3 install --upgrade setuptools

# Install required dependencies
pip3 install numpy==1.17.5

pip3 install pandas
pip3 install scikit-image

pip3 install torch
pip3 install torchvision

pip3 install ipykernel
pip3 install matplotlib

# Setup COCO's dataset and API
source download_coco_dataset.sh
source setup_coco_api.sh

# Preprocess COCO's dataset
python3 karpathy_split.py
python3 build_vocab.py

# Resize COCO's images to 224x224
python3 resize.py

# Keep only resized COCO's images
rm -rf ./data/train2014
rm -rf ./data/val2014

# Keep only COCO's Karpathy splits
rm ./data/annotations/caption*.json
