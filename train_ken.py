import good_read_data as rd
from function import *
from model import *
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from time import time
import datetime

""" Limit memory """
# Auto
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

img_shape = (512, 512)

day = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%Y-%m-%d %H:%M")
startTime = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%H:%M")
# Get data
print("> Loading data..")
#x_train, y_train = rd.read_dataset("./dataset/pomelo_seg_n/train/*", 0)
#x_test, y_test = rd.read_dataset("./dataset/pomelo_seg_n/test/*", 0)
x_train, y_train = rd.read_dataset("./dataset/original/train/*", 0, img_shape)
x_test, y_test = rd.read_dataset("./dataset/original/test/*", 0, img_shape)
print("> Done")
print(y_train.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)

# Build model
input_shape = (512, 512, 3)
num_class = 4
final_Act = 'sigmoid'

batch_size = 10
epoch = 90

model = densenet(input_shape, num_class, final_Act)
#model = alexnet
#opt = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)

opt = SGD(lr=0.01)#, decay=0.0001, momentum=0.9, nesterov=True)
#opt = Adam(lr=0.0001)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["categorical_accuracy"])
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=10, mode='auto', cooldown=3, min_lr=0.00001)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step_1 = epoch//3
    decay_step_2 = (epoch//3)*2
    if epoch % decay_step_1 == 0 and epoch:
        return lr * decay_rate
    if epoch % decay_step_2 == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [
    reduce_lr
    #LearningRateScheduler(lr_scheduler, verbose=1)
]



train_history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_val, y_val))

endTrain = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%H:%M")
y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
label = np.argmax(y_val, axis=1)

endTest = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%H:%M")

"""
y_pred = model.predict(x_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
np.set_printoptions(suppress=True)

trueLabel = np.argmax(y_test, axis=1)
#labels = ["black", "healthy", "mg", "moth", "oil"]
labels = ["black", "mg", "moth", "oil"]
print(classification_report(trueLabel, pred, target_names=labels))
print('Accuracy:  ',accuracy_score(trueLabel,pred))
print(confusion_matrix(trueLabel, pred))
acc = accuracy_score(trueLabel,pred)
model.save(f"model/5c_{round(acc, 4)*100}.h5")
"""

# Testing on testing data
threshold = 0.2
print("\nTesting on testing data...")
y_pred = model.predict(x_test)
pred = list()
for i in range(len(y_pred)):
    np.set_printoptions(formatter=dict(float=lambda t:"%3.7f" % t))
    print("Sample ", i)
    print("> Pred: ", y_pred[i])
    print("> True: ", y_test[i])
    c = 0
    for a in y_pred[i]:
        if a < threshold:
            c = c+1
    if c == 4:
        pred.append(0)
        continue
    pred.append(np.argmax(y_pred[i])+1)
    
label = list()
for i in range(len(y_test)):
    #print(i)
    c = 0
    for a in y_test[i]:
        if a == 0:
            c = c+1
    if c == 4:
        label.append(0)
        continue
    label.append(np.argmax(y_test[i])+1)

acc = accuracy_score(label,pred)

scores = model.evaluate(x_test, y_test, verbose=0)

print('\nTesting result:')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test normal accuracy:', acc)
print("0 is healthy\n")
labels = ["black", "mg", "moth", "oil"]
#labels = ["black", "healthy", "mg", "moth", "oil"]
labels = ["healthy", "black", "mg", "moth", "oil"]
trueLabel = np.argmax(y_test, axis=1)

print(classification_report(label, pred, target_names=labels))
print('Accuracy:  ',accuracy_score(label,pred))
print(confusion_matrix(label, pred))
# Save model
model.save(f'model/densenet_ml0915.h5')
print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)




f = open('record/New Training Record.txt', 'a+')
f.write(day)
f.write("\n")
f.write(f"Epochs:     {epoch}\nBatch_size: {batch_size}\nModel:      DenseNet\nAccuracy:   {acc}\n")
f.write(f"Training:   start at {startTime}, end at {endTrain}\nTesting:    start at {endTrain}, end at {endTest}")
f.write("\n======================================================\n")

""" Training result """
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