import argparse
import os

from PIL import Image

def resize_image(image, size):
    """Resize an image to the specified size. If you try to resize without a filter, a phenomenon called aliasing typically manifests as obnoxious pixellated effects. Therefore, we specify an anti-aliasing filter, like Image.ANTIALIAS, when resizing."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize all the images in 'image_dir' to the specified size and save them into 'output_dir'. Since no sanity checks are performed, we are of course expecting 'image_dir' to contain images and only images suitable for this purpose."""
    # Retrieve all the image filenames from the input dir
    images = os.listdir(image_dir)
    num_images = len(images)

    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f: # Append the image filename to the input dir path,
                                                               # in order to open it up with PIL
            with Image.open(f) as img:
                # Resize the image and save it in the output dir
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
                
        if i % 100 == 0:
            print ("[%d/%d] Resized images and saved into '%s'."
                   %(i, num_images, output_dir))

def main(args):
    """Resize all COCO's training images to the desired size, in order to match the requirements of the encoder's input in our Encoder2Decoder framework from models.py. Since we are using ResNet152 as a CNN encoder there, images must be resized to 224x224."""
    splits = ['train', 'val']
    years  = ['2014'] # COCO's datasets year
    
    # If the image output path directory does not exist, create it.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    for split in splits:
        for year in years:
            # Build path for input and output dataset
            dataset = split + year

            # (i.e., image_dir/val2014)
            image_dir  = os.path.join(args.image_dir,  dataset)
            # (i.e., image_dir/resized/val2014)
            output_dir = os.path.join(args.output_dir, dataset)
            
            # Resize each image to a square
            # (args.image_size x args.image_size)
            image_size = [
                args.image_size,
                args.image_size
            ]

            resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths to directories
    parser.add_argument('--image_dir', type=str, default='./data',
                        help='Directory with train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized',
                        help='Directory to save resized images to')
    
    # Resizing hyperparameters
    parser.add_argument('--image_size', type=int, default=224,
                        help='Images size after resizing')

    # Parse arguments and run
    args = parser.parse_args()
    main(args)
