from PIL import Image
import os, sys

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

dataset_dir = "./dataset/pomelo_multi_aug/train"
class_list = os.listdir(dataset_dir)
base_dir = './dataset/pomelo'
#os.mkdir(base_dir)
target_dir = os.path.join(base_dir, 'train')
os.mkdir(target_dir)

print("Class list:", class_list)

for cls in class_list:
    if cls == ".DS_Store":
        continue
    print("#Making folder", os.path.join(target_dir, cls))
    os.mkdir(os.path.join(target_dir, cls))
    path = os.path.join(dataset_dir, cls)
    fnames = os.listdir(path)
    #print(fnames)
    for f in fnames:
        if f == ".DS_Store":
            continue
        img = Image.open(os.path.join(path, f))
        img = exif_transpose(img)
        img.convert('RGB')
        img = img.resize((256,256))
        print(" > Saving file", os.path.join(os.path.join(target_dir, cls), f))
        img.save(os.path.join(os.path.join(target_dir, cls), f))

