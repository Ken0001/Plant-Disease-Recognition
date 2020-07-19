import read_data as rd
from model import *
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Multi-task learning based on densenet
"""

# Read training and testing data
#
# def read_dataset(loc, augment mode):
#     return img, label

x_train, y_train = rd.read_dataset("./dataset/new/train/*", 1)
x_test, y_test = rd.read_dataset("./dataset/new/test_un/*", 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)
print("/n")
print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)
print("Done!")

# Prepare Model
# In model.py
# densenet, densenet_multi, cnn, alexnet, vgg

img_shape = (224, 224, 3)
num_class = 4
final_Act = 'softmax'

batch_size = 16
epoch = 90

model = densenet(img_shape, num_class, final_Act)
opt = SGD(lr=0.01)#, decay=0.0001, momentum=0.9, nesterov=True)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["categorical_accuracy"])
# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5, mode='auto', cooldown=3, min_lr=0.000001)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    if epoch % 30 == 0 and epoch:
        return lr * decay_rate
    if epoch % 60 == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [
    #reduce_lr
    LearningRateScheduler(lr_scheduler, verbose=1)
]

# Training
train_history = model.fit(x_train, y_train,
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
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
label = np.argmax(y_test, axis=1)
acc = accuracy_score(label,pred)

scores = model.evaluate(x_test, y_test, verbose=0)

print('\nEvaluate result:')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test normal accuracy:', acc)

labels = ["black", "mg", "moth", "oil"]
#labels = ["black", "healthy", "mg", "moth", "oil"]
trueLabel = np.argmax(y_test, axis=1)

print(classification_report(trueLabel, pred, target_names=labels))
print('Accuracy:  ',accuracy_score(trueLabel,pred))
print(confusion_matrix(trueLabel, pred))
# Save model
model.save(f'model/densenet.h5')
print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)

# Show result
print(train_history.history.keys())
acc = train_history.history['categorical_accuracy']
val_acc = train_history.history['val_categorical_accuracy']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
p_epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(p_epochs, acc, 'b', label='Training accurarcy')
plt.plot(p_epochs, val_acc, 'g', label='Validation accurarcy')
plt.plot(p_epochs, loss, 'r', label='Training loss')
plt.plot(p_epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and Validation')
plt.show()