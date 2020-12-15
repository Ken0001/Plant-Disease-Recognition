#import good_read_data as rd
import good_read_data_ml5 as rd
from model import *
from ML_DenseNet import mldensenet
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot
#%matplotlib inline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
"""
Multi-task learning based on densenet
"""

# Read training and testing data
#
# def read_dataset(loc, augment mode):
#     return img, label

x_train, y_train = rd.read_dataset("./dataset/pomelo_ml5/train/*", 1)
x_test, y_test = rd.read_dataset("./dataset/pomelo_ml5/test/*", 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)
print("/n")
print(y_test)
print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)
print("Done!")

#print(rd.num_instance)
# Prepare Model
# In model.py
# densenet, densenet_multi, cnn, alexnet, vgg

img_shape = (224, 224, 3)
num_class = 5
final_Act = 'sigmoid'

batch_size = 16
epoch = 100

#model = mldensenet(img_shape, num_class, mltype=5, finalAct=final_Act)
model = densenet_ml5(img_shape, num_class, final_Act)
opt = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
#opt = Adam(lr=0.001)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["binary_accuracy", "categorical_accuracy"])
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["categorical_accuracy"])
# Callbacks
#reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5, mode='auto', cooldown=3, min_lr=0.000001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=10, mode='auto', cooldown=3, min_lr=0.00001)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    if epoch % 30 == 0 and epoch:
        return lr * decay_rate
    if epoch % 60 == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [
    reduce_lr
    #LearningRateScheduler(lr_scheduler, verbose=1)
]

# Training
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          callbacks=callbacks,
          shuffle=True,
          validation_data=(x_val, y_val))

# Testing on validation data
print("\nTesting on validation data...\n")
y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
label = np.argmax(y_val, axis=1)
acc = accuracy_score(label,pred)

scores = model.evaluate(x_val, y_val, verbose=0)

print('\nEvaluate result:')
print('Val loss:', scores[0])
print('Val accuracy:', scores[1])
print('Val normal accuracy:', acc)

# Testing on testing data
print("\nTesting on testing data...")
y_pred = model.predict(x_test)
threshold = 0.4
pred = np.zeros(np.array(y_test).shape, dtype=np.int)
np.set_printoptions(precision=6, suppress=True)
for i in range(len(y_pred)):
    for j in range(5):
        if y_pred[i][j] >= threshold:
            pred[i][j] = 1
        else:
            pred[i][j] = 0
    print("Sample ", i)
    print("> Pred: ", y_pred[i])
    print("> Pred: ", pred[i])
    print("> True: ", y_test[i])

true = np.array(y_test)
acc = accuracy_score(true, pred)
scores = model.evaluate(x_test, y_test, verbose=0)

print('\nEvaluate result:')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test normal accuracy:', acc)

sample = 0
score = 0
for i in range(len(pred)):
    sample += 1
    #print(pred[i])
    #print(true[i])
    #print("--------")
    sample_acc = accuracy_score(true[i], pred[i])
    #print("acc=",sample_acc)
    score += sample_acc
    #print("--------")
    # for j in range(4):
    #     print(j)
print("\nMy Evaluate")
print("Sample:",sample)
print("Accuracy:",acc)
print("Precision:", score/sample)