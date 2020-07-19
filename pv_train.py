from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from model import *
from tensorflow.keras.optimizers import Adam, SGD, Adadelta

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_shape = (150, 150, 3)
num_class = 39
final_Act = 'softmax'


model = cnn(img_shape, num_class, final_Act)
opt = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
#opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["categorical_accuracy"])

train_dir = "./dataset/pv80-20//train"
validation_dir = "./dataset/pv80-20//validation"

# Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir, # Target directory
    target_size=(150, 150), #Resizes all images to 150 × 150
    batch_size=64,
    class_mode='categorical') #Because you use binary_crossentropy loss, you need binary labels


validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')


reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5, mode='auto', cooldown=3, min_lr=0.000001)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    if epoch % 10 == 0 and epoch:
        return lr * decay_rate
    if epoch % 20 == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [
    #reduce_lr
    #LearningRateScheduler(lr_scheduler, verbose=1)
]

print("Start train")
history = model.fit_generator(
    train_generator,
    #steps_per_epoch=230,
    epochs=60,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=43)

test_dir = "./dataset/pv80-20//test"

test_datagen = ImageDataGenerator(rescale=1./255)


test_generator = test_datagen.flow_from_directory(
    test_dir, # Target directory
    target_size=(150, 150), #Resizes all images to 150 × 150
    batch_size=64,
    class_mode='categorical') #Because you use binary_crossentropy loss, you need binary labels


score = model.evaluate_generator(test_generator, steps=40, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

print("Testing loss:", score[0])
print("Testing accuracy:", score[1])

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
print(history.history.keys())
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
p_epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(p_epochs, acc, 'b', label='Training accurarcy')
plt.plot(p_epochs, val_acc, 'g', label='Validation accurarcy')
plt.plot(p_epochs, loss, 'r', label='Training loss')
plt.plot(p_epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and Validation')

plt.savefig("ndnrecord.png")
plt.show()
