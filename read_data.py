import os, sys
import glob 
import numpy as np
from PIL import Image, ImageFile
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import PIL.Image
import PIL.ImageOps

#ImageFile.LOAD_TRUNCATED_IMAGES = True

train=[]
labels_hot=[]
dict_labels = {"black":0, "healthy":1, "mg":2, "moth":3, "oil":4}

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


def load_image_file(file, mode='RGB', size=None):
    # Load the image with PIL
    img = PIL.Image.open(file)
    #img = exif_transpose(img)
    img = img.convert(mode)
    if size:
        if type(size) is not tuple:
            print("Wrong type of size")
        else:
            img = img.resize(size)
    return img

def aug(img, size, label, mode=1):
    padding = int((img.size[0]-size[0])/2)
    left = padding
    top = padding
    right = img.size[0]-padding
    down = img.size[1]-padding
    crop_loc=(left, top, right, down)
    shift = 10
    if mode==1 or mode ==0:
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
    if mode==5:
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left-shift, top, right-shift, down)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left, top-shift, right, down-shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left+shift, top, right+shift, down)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left, top+shift, right, down+shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
    if mode==9:
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left-shift, top, right-shift, down)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left, top-shift, right, down-shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left+shift, top, right+shift, down)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left, top+shift, right, down+shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left-shift, top-shift, right-shift, down-shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left-shift, top+shift, right-shift, down+shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left+shift, top-shift, right+shift, down-shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])
        crop_loc=(left+shift, top+shift, right+shift, down+shift)
        train.append(np.array(img.crop(crop_loc)))
        labels_hot.append(dict_labels[label])

def read_dataset(location, mode=1, input_shape = (224,224)):
    #print(location)
    #print(mode)
    global train, labels_hot
    for folders in glob.glob(location):
        #print(folders)
        for filename in os.listdir(folders):
            if filename == ".DS_Store":
                continue
            label = os.path.basename(folders)
            className = np.asarray(label)
            print(f"\r{os.path.join(folders,filename)}", end="")
            sys.stdout.flush()
            img = load_image_file(os.path.join(folders,filename), size=(256,256))
            aug(img, (224,224), label, mode)
            

    x_train = np.array(train, dtype=np.float16) / 255.0
    y_train = to_categorical(LabelEncoder().fit_transform(labels_hot))
    train=[]
    labels_hot=[]
    return x_train, y_train