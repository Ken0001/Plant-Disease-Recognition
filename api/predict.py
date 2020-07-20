import pandas as pd
import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array
#from keras import backend as K
#K.clear_session()

label_list = ["黑點病","健康","缺鎂","潛葉蛾","油胞病"]

print("> Preparing Model...")
model = load_model("densenet.h5")
print("> Model loaded")

def convert_image_to_array(image_dir, width, height):
    print("Convert")
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            if (image.shape[0]<image.shape[1]):
                image = np.rot90(image)
            image = cv2.resize(image, (width,height)) #size:(w,h)
            #print(image/255)
            return img_to_array(image)
        else :
            print("wtgf")
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def toPanda(y_pred, pred, target_image):
    prnit("pandas")
    result = np.hsplit(y_pred, 5)
    Black = result[0].flatten()
    Healthy = result[1].flatten()
    Mg = result[2].flatten()
    Moth = result[3].flatten()
    Oil = result[4].flatten()

    c = ['Image', 'Prediction', 'Black', 'Health', 'Mg', 'Moth', 'Oil']
    d = [target_image,label_list[pred[0]], Black, Healthy, Mg, Moth, Oil]
    record = pd.Series(d, index=c)

    return record

#def loadModel(trained_model):
#    model = load_model(trained_model)
#    return model

def loadImg(target_image):
    print("> Preparing Images...")
    width, height = 224, 224
    img = []
    img.append(convert_image_to_array(target_image, width, height))
    img = np.array(img, dtype=np.float16) / 255.0

    return img


def predict(img, model):
    print("> Starting Predict")
    model.summary()
    y_pred = model.predict(img)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #np.set_printoptions(suppress=True)
    result = toPanda(y_pred, pred, target_image)

    return result

def api(target_image, trained_model):
    #model = loadModel(trained_model)
    img = loadImg(target_image)
    #result = predict(img, model)
    y_pred = model.predict(img)
    print (y_pred)
    result = "0"
    return result


