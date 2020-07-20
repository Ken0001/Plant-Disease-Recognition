# -*- coding: utf-8 -*-
import os
from flask import Flask, request
from werkzeug import secure_filename

# DL
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from func import loadImg, toPanda


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
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

def makeResult(r):
    result = "辨識結果:  "+ str(r.Prediction) + ",\
        黑點病:"+ str(round(r.Black[0]*100,4)) + "%,\
        缺鎂:"+ str(round(r.Mg[0]*100,4)) + "%,\
        潛葉蛾:"+ str(round(r.Moth[0]*100,4)) + "%,\
        油胞病:"+ str(round(r.Oil[0]*100,4)) + "%,\
        健康:"+ str(round(r.Health[0]*100,4)) + "%"

    return result

""" Return format 
辨識結果:xxx,黑點病:xxx,...

JAVA: split with ","
"""
def make_html_result(r):
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
    <title></title>
    <style type="text/css">
        .result-box {
            padding-bottom: 10px;
            padding-top: 10px;
        }

        .result-box div {
            padding: 5px 40px;
            background: #E0E0E0;
        }

        .result-box div:nth-child(2n) {
            background: #BDBDBD;
        }

        .result-box div.result-main {
            background: #455A64;
            margin-top: -15px;
            margin-bottom: 30px;
            color: #fff;
            text-align: center;
        }
    </style>
    </head>
    <body>
    </body>
    </html>
    '''

    result = "<div class='result-box'><br><div class='result-main'><span class='result-value'>" + str(r.Prediction) + "</span></div> \
        <div><span class='result-title'>黑點病</span><span class='result-value'>" + str(round(r.Black[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>缺鎂</span><span class='result-value'>" + str(round(r.Mg[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>潛葉蛾</span><span class='result-value'>" + str(round(r.Moth[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>油胞病</span><span class='result-value'>" + str(round(r.Oil[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>健康</span><span class='result-value'>" + str(round(r.Health[0]*100,4)) + "%</span></div></div>"
    
    return html + result


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def api():
    if request.method == 'POST':
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
            res = makeResult(r)
            return res
        else :
            return "不支援\""+ file.filename.split('.')[1] +"\"檔案格式"
        print(file)
        return "上傳成功"

    return "沒傳東西吧?"

if __name__ == '__main__':
    loadModel()
    app.run(host="0.0.0.0", port="7789", debug=True, threaded=True)