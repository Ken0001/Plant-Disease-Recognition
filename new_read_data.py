import os, sys
import glob 
import numpy as np
from PIL import Image

import PIL.Image
import PIL.ImageOps

dict_labels = {"black":0, "mg":1, "moth":2, "oil":3}
image, labels = [],[]

# Get the code of label (like [0, 1, 0, 0])
def label_encoder(name, multi=False):
    global labels
    if multi==True:
        lst = [0,0,0,0]
        for i in range(len(lst)):
            for check in name.split("_")[:-1]:
                if i == dict_labels[check]:
                    lst[i] = 1
        labels.append(lst)
    else:
        if name == "healthy":
            labels.append([0,0,0,0])
        elif name == "black":
            labels.append([1,0,0,0])
        elif name == "mg":
            labels.append([0,1,0,0])
        elif name == "moth":
            labels.append([0,0,1,0])
        elif name == "oil":
            labels.append([0,0,0,1])

# Handle the exif info of picture
def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

# Get image
def load_image_file(file, mode='RGB', size=None):
    # Load the image with PIL
    img = PIL.Image.open(file)
    img = exif_transpose(img)
    img = img.convert(mode)
    if size:
        if type(size) is not tuple:
            print("Wrong type of size")
        else:
            img = img.resize(size)
    return img
    
# Main function
def read_dataset(loc, aug=False, input_shape=(224,224)):
    """
    print(f"\r{os.path.join(folders,filename)}", end="")
    """
    global image, labels
    for folders in glob.glob(loc):
        for file in os.listdir(folders):
            print(f"\rLoading data: {os.path.join(folders,file)}", end="")
            if os.path.basename(folders) == "multi":
                label_encoder(file, True)
                img = load_image_file(os.path.join(folders,file), size = (224,224))
                image.append(np.array(img))
            else:
                label_encoder(os.path.basename(folders), False)
                img = load_image_file(os.path.join(folders,file), size = (224,224))
                image.append(np.array(img))
    
    x = np.array(image, dtype=np.float16) / 255.0
    y = np.array(labels, dtype=np.float16)
    image, labels = [],[]
    return x, y