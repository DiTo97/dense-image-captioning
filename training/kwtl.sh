#!/bin/bash

# 'Knowing when to Look' adaptive model training
# https://arxiv.org/pdf/1612.01887.pdf

# Note: This script assumes that you've already run preprocessing.sh and that a preprocessing.zip file, containing COCO's API and data, is available at the same level as well as all the other necessary Python files that are addressed within train.py.

# Clean old Python venv
rm -rf ./CV-env

# Create Python venv on CERN's sandbox
python3 -m venv CV-env
source CV-env/bin/activate

# Update PIP manager
pip3 install --upgrade pip
pip3 install --upgrade setuptools

# Install required dependencies
pip3 install numpy==1.17.5
pip3 install gensim
pip3 install pandas
pip3 install scikit-image
pip3 install pyemd

pip3 install --upgrade Pillow

pip3 install torch
pip3 install torchvision

pip3 install ipykernel
pip3 install matplotlib

unzip preprocessing.zip

# Move COCO's API and data up one level
mv content/* .

# Remove leftover files
rm preprocessing.zip
rm -rf content

python3 train.py --num_epochs=25 --pretrained='adaptive-?.pkl' # Choose a pre-trained model to start from,
                                                               # otherwise take the --pretrained flag out
