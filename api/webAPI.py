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

from flask_cors import CORS 

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

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
            img = loadImg(image)
            with graph.as_default():
                y_pred = model.predict(img)
            
            pred = list()
            for i in range(len(y_pred)):
                pred.append(np.argmax(y_pred[i]))
            r = toPanda(y_pred, pred, image)
            res = make_result(r)
            # file_url => /uploads/xxx.jpg
            file_url = url_for('uploaded_file', filename=filename)
            
            return res
    
    return "A"



if __name__ == '__main__':
    print("> Loading...")
    loadModel()
    app.run(host="0.0.0.0", port="7788", debug=True)
