from PIL import Image, ImageOps
import numpy as np
import os,glob
from tqdm import tqdm

img_path="./dataset/pomelo_seg/train/*"
aug_path="./dataset/pomelo_seg_aug/train/"

# Shift, Mirror, and Rotate
def shift(img, filename, folder):
    print("\nSHIFT")
    size = (224,224)
    padding = int((img.size[0]-size[0])/2)
    left = padding
    top = padding
    right = img.size[0]-padding
    down = img.size[1]-padding
    shift = 10
    dict = {
        0:(left, top, right, down),
        1:(left-shift, top, right-shift, down),
        2:(left, top-shift, right, down-shift),
        3:(left+shift, top, right+shift, down),
        4:(left, top+shift, right, down+shift),
        5:(left-shift, top-shift, right-shift, down-shift),
        6:(left-shift, top+shift, right-shift, down+shift),
        7:(left+shift, top-shift, right+shift, down-shift),
        8:(left+shift, top+shift, right+shift, down+shift),
    }
    for i in range(5):
        target = aug_path+folder+"/"+filename.split(".")[0]+"S"+str(i)+".jpg"
        print(target)
        aug_img = img.crop(dict[i])
        aug_img.save(target)

def mirror(img, filename, folder):
    print("\nMIRROR")
    target = aug_path+folder+"/"+filename.split(".")[0]+"M.jpg"
    img = img.resize((224, 224))
    aug_img = ImageOps.mirror(img)
    print(target)
    aug_img.save(target)

def rotate(img, filename, folder):
    print("\nROTATE")
    deg = (45,90,135,180,225,270,315)
    img = img.resize((224, 224))
    for i in deg:
        target = aug_path+folder+"/"+filename.split(".")[0]+"R"+str(i)+".jpg"
        aug_img = img.rotate(i)
        print(target)
        aug_img.save(target)


for folders in glob.glob(img_path):
    print(f"Loading {folders}:")
    for f in tqdm(os.listdir(folders)):
        f_name = os.path.join(folders,f)
        print(f"\rLoading data: {f_name}", end="")
        img = Image.open(f_name)
        img = img.convert("RGB")
        shift(img, f, os.path.basename(folders))
        mirror(img, f, os.path.basename(folders))
        rotate(img, f, os.path.basename(folders))
