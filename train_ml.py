import new_read_data as rd
from model import *
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

""" Limit memory """
# Auto
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

print("Training data:")
#x_train, y_train = rd.read_dataset("./dataset/new/train/*", 0)
# augment: 0 or 5 or 9
x_train, y_train = rd.read_dataset("./dataset/new/train_b/*", 9)
print("\n")
print(x_train.shape, y_train.shape)
print("\nTesting data:")
x_test, y_test = rd.read_dataset("./dataset/new/test/*", 0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)

input_shape = (224, 224, 3)
num_classes = 4
finalAct = 'sigmoid'

batch_size = 16
epochs = 90

model = densenet_multi(input_shape, num_classes, finalAct)
opt = SGD(lr=0.01)#, decay=0.0001, momentum=0.9, nesterov=True)
#opt = Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=5, mode='auto', cooldown=3, min_lr=0.000001)

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

train_history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          shuffle=True,
          validation_data=(x_val, y_val))

print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)

y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
label = np.argmax(y_val, axis=1)
acc = accuracy_score(label,pred)

model.save(f"model/densenet_multi_task.h5")



scores = model.evaluate(x_val, y_val)
print('Val loss:', scores[0])
print('Val accuracy:', scores[1])

scores_t = model.evaluate(x_test, y_test)
print('Test loss:', scores_t[0])
print('Test accuracy:', scores_t[1])

labels = ["black", "mg", "moth", "oil"]
#labels = ["black", "healthy", "mg", "moth", "oil"]
y_pred = model.predict(x_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
np.set_printoptions(suppress=True)

trueLabel = np.argmax(y_test, axis=1)

print(classification_report(trueLabel, pred, target_names=labels))
print('Accuracy:  ',accuracy_score(trueLabel,pred))
print(confusion_matrix(trueLabel, pred))

print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)

""" Training result """
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
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