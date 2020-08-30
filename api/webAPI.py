# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename
#from predict import predict
#import predict
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from func import loadImg, toPanda

# Image
from PIL import Image

from flask_cors import CORS

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'jpeg', 'gif', 'HEIC'])

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = os.getcwd()+"/upload_img"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 自動增長 GPU 記憶體用量
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 Session
tf.keras.backend.set_session(sess)


model = None

def loadModel():
    global model
    global graph
    model = load_model("densenet.h5")
    graph = tf.get_default_graph()


def load_image_file(file, mode='RGB', size=None):
    # Load the image with PIL
    img = Image.open(file)
    img = exif_transpose(img)
    img = img.convert(mode)
    if size:
        if type(size) is not tuple:
            print("Wrong type of size")
        else:
            img = img.resize(size)
    return img

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
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def make_result(r):
    
    result = "<h4 class='result-value'>" + str(r.Prediction) + "</h4> \
        <table class='striped centered'><tbody><tr><td class='result-title'>黑點病</td><td class='result-value'>" + str(round(r.Black[0]*100,4)) + "%</td></tr> \
        <tr><td class='result-title'>缺鎂</td><td class='result-value'>" + str(round(r.Mg[0]*100,4)) + "%</td></tr> \
        <tr><td class='result-title'>潛葉蛾</td><td class='result-value'>" + str(round(r.Moth[0]*100,4)) + "%</td></tr> \
        <tr><td class='result-title'>油胞病</td><td class='result-value'>" + str(round(r.Oil[0]*100,4)) + "%</td></tr> \
        <tr><td class='result-title'>健康</td><td class='result-value'>" + str(round(r.Health[0]*100,4)) + "%</td></tr></tbody></table>"
    
    return result
    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # get the file
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            #img = loadImg(imgPath)
            x = []
            img = load_image_file(imgPath, size=(224, 224))
            x.append(np.array(img))
            x = np.array(x, dtype=np.float16) / 255.0
            with graph.as_default():
                y_pred = model.predict(x)
            
            pred = list()
            for i in range(len(y_pred)):
                pred.append(np.argmax(y_pred[i]))
            r = toPanda(y_pred, pred, imgPath)
            res = make_result(r)
            # file_url => /uploads/xxx.jpg
            file_url = url_for('uploaded_file', filename=filename)
            
            return res
    
    return "A"



if __name__ == '__main__':
    print("> Loading...")
    loadModel()
    app.run(host="0.0.0.0", port="7788", debug=True)
