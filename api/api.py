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

html = '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>柚子病蟲害辨識</title>
    <meta name="viewport" content="width=device-width, user-scalable=no">
    <link rel="icon" type="image/png" href="https://imgur.com/TIE6Hcl.png">
    <script src="https://code.jquery.com/jquery-3.4.1.min.js" integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/exif-js@2.3.0/exif.min.js"></script>
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css?family=Varela+Round&display=swap');
        @import url('https://fonts.googleapis.com/css?family=Noto+Sans+TC&display=swap');

        html, body {
            min-height: 100vh;
            font-family: 'Noto Sans TC', sans-serif;
        }

        body {
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        .box {
            margin-top: 30px;
            text-align: center;

        }

        label {
            color: #f1e5e6;
            max-width: 250px;
            border: 3px solid #d3394c;
            color: #d3394c;
            font-size: 1.25rem;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
            display: inline-block;
            overflow: hidden;
            padding: 0.625rem 1.25rem;
            font-family: 'Varela Round', sans-serif;
        }
        label:hover {
            border-color: #b71c1c;
            color: #b71c1c;
        }
        svg {
            width: 1em;
            height: 1em;
            vertical-align: middle;
            fill: currentColor;
            margin-top: -0.25em;
            margin-right: 0.25em;
        }

        .img-box {
            text-align: center;
        }
        
        .img-box img {
            max-height: 40vh;
            width: auto;
        }

        @media screen and (max-width: 425px) {
            .img-box img {
                max-width: 80vw;
                max-height: unset;
                height: auto;
            }
        }

        #pop-up-page {
            width: 100%;
            height: 0;
            position: absolute;
            margin: auto;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,.5);
            overflow: hidden;
            /*transition: height .3s linear;
            -webkit-transition: height 0.3s linear;*/
            display: flex;
            align-items: center;
            z-index: 9;
        }

        #pop-up-page.pop-up-on {
            height: 100vh;
        }

        .loading {
            display: inline-block;
            position: relative;
            width: 360px;
            height: 360px;
            margin: auto;
        }

        .loading div {
            box-sizing: border-box;
            display: block;
            position: absolute;
            width: 344px;
            height: 344px;
            margin: 8px;
            border: 12px solid #40C4FF;
            border-radius: 50%;
            animation: loading 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
            border-color: #40C4FF transparent transparent transparent;
        }

        .loading div:nth-child(1) {
        animation-delay: -0.45s;
        }

        .loading div:nth-child(2) {
        animation-delay: -0.3s;
        }

        .loading div:nth-child(3) {
        animation-delay: -0.15s;
        }

        @keyframes loading {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }

        button.btn {
            border: 0;
            background: #d3394c;
            color: #fff;
            font-family: 'Varela Round', sans-serif;
            padding: 5px 8px 5px 10px;
            font-size: 16px;
            letter-spacing: 2px;
            margin-top: 10px;
        }

        button.btn:hover {
            background: #b71c1c;
        }

        button.btn:focus {
            outline: none;
        }

        span.result-title {
            width: 100px;
            display: inline-block;
        }

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

        .rotate90 {
            transform: rotate(90deg) scale(0.8, 0.8);
        }
        .rotate270 {
            transform: rotate(270deg) scale(0.8, 0.8);
        }
        .rotate180 {
            transform: rotate(180deg);
        }
    </style>
    </head>
    <body>
    <div id="pop-up-page" class="">
        <div class="loading"><div></div><div></div><div></div><div></div></div>
    </div>
    
    <div class="box">
        <h1>柚子病蟲害辨識系統</h1>
        <form method="POST" enctype="multipart/form-data" id="nameform">
                <input type="file" style="display:none;" name="file" id="file" class="inputfile" required="required">
                <label for="file"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="17" viewBox="0 0 20 17"><path d="M10 0l-5.2 4.9h3.3v5.1h3.8v-5.1h3.3l-5.2-4.9zm9.3 11.5l-3.2-2.1h-2l3.4 2.6h-3.5c-.1 0-.2.1-.2.1l-.8 2.3h-6l-.8-2.2c-.1-.1-.1-.2-.2-.2h-3.6l3.4-2.6h-2l-3.2 2.1c-.4.3-.7 1-.6 1.5l.6 3.1c.1.5.7.9 1.2.9h16.3c.6 0 1.1-.4 1.3-.9l.6-3.1c.1-.5-.2-1.2-.7-1.5z"></path></svg> <span>選擇檔案</span></label>
        </form>
        <button type="submit" form="nameform" value="Submit" class="btn">上傳</button>
	</div>

    <script>
        const fileUploader = document.querySelector('#file');

        fileUploader.addEventListener('change', (e) => {
            $('label span').text(e.target.files[0].name);
        });

        $('.btn').on('click', function() {
            if ($('label span').text() != "Choose a file"){
                $('#pop-up-page').addClass('pop-up-on');
            }
        });     
    </script>
    </body>
    </html>
    '''

exif = '''
        <script>
            var tar = document.getElementById('target');
            function getExif() {
                EXIF.getData(tar, function() {
                        var orientation = EXIF.getTag(this, "Orientation");
                        console.log(orientation)
                        if(orientation == 6) {
                            this.className = "rotate90";
                        } else if(orientation == 8) {
                            this.className = "rotate270";
                        } else if(orientation == 3) {
                            this.className = "rotate180";
                        }
                });
            };
            getExif();
        </script>
    '''

def make_result(r):
    
    result = "<div class='result-box'><br><div class='result-main'><span class='result-value'>" + str(r.Prediction) + "</span></div> \
        <div><span class='result-title'>黑點病</span><span class='result-value'>" + str(round(r.Black[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>缺鎂</span><span class='result-value'>" + str(round(r.Mg[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>潛葉蛾</span><span class='result-value'>" + str(round(r.Moth[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>油胞病</span><span class='result-value'>" + str(round(r.Oil[0]*100,4)) + "%</span></div> \
        <div><span class='result-title'>健康</span><span class='result-value'>" + str(round(r.Health[0]*100,4)) + "%</span></div></div>"
    
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
            
            return html + '<br><div class="img-box"><img src=' + file_url + ' id = "target"></div>' + res + exif
    return html



if __name__ == '__main__':
    print("> Loading...")
    loadModel()
    app.run(host="0.0.0.0", port="7788", debug=True)
