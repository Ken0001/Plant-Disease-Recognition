# Model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, BatchNormalization, Concatenate, Add, ReLU, Activation
import keras.backend as K


# DenseNet
# 6,12,24,16 - 121
# 6,12,32,32 - 169
# 6,12,48,32 - 201
# 6,12,64,48 - 264
def densenet(img_shape, n_classes, finalAct='softmax', f = 32):
    repetitions = 6,12,24,16

    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(x, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        #x = Dropout(0.2)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 16, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x = Conv2D(64, 7, strides=2, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
    
    x = GlobalAvgPool2D()(d)
    
    output = Dense(n_classes, activation=finalAct)(x)
    
    model = Model(input, output)
    
    return model

def densenet264(img_shape, n_classes, finalAct='softmax', f = 32):
    repetitions = 6,12,48,32

    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(x, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = Dropout(0.5)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
    
    x = GlobalAvgPool2D()(d)
    
    output = Dense(n_classes, activation=finalAct)(x)
    
    model = Model(input, output)
    
    return model

# Multiple branch based on DenseNet
def densenet_ml1(img_shape, n_classes, finalAct='softmax', f = 32):
    # repetitions = 6, 12, 24, 16
    repetitions = 6, 12, 24
    r2 = 16
    
    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(tensor, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        #x = Dropout(0.5)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    """ Testing
    r = 3
    d = dense_block(x, r)
    x = transition_block(d)
    """
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
    
    outputs = []
    for i in range(n_classes):
        print("class ", i)
        d = dense_block(x, r2)
        branch = transition_block(d)
        branch = GlobalAvgPool2D()(d)
        output = Dense(1, activation=finalAct)(branch)
        outputs.append(output)
        
    outputs = Concatenate()(outputs)
    """ Example on Resnet V2
    outputs = []
    for i in range(n_classes):
        output = output_layer(x,num_filters_in)
        outputs.append(output)
    # concate for output purpose
    outputs = keras.layers.Concatenate()(outputs)
    """
    """ Original
    x = GlobalAvgPool2D()(d)
    
    output = Dense(n_classes, activation=finalAct)(x)
    """
    model = Model(input, outputs)
    
    return model

def densenet_ml2(img_shape, n_classes, finalAct='softmax', f = 32):
    repetitions = 6,12,24,16

    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(tensor, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        #x = Dropout(0.3)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
    
    x = GlobalAvgPool2D()(d)
    
    outputs = []
    for i in range(n_classes):
        print("\rclass ", i, end="")
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation=finalAct)(x)
        outputs.append(output)

    outputs = Concatenate()(outputs)
    #output = Dense(n_classes, activation=finalAct)(x)
    
    model = Model(input, outputs)
    
    return model

def densenet_ml3(img_shape, n_classes, finalAct='softmax', f = 32):
    repetitions = 6,12,24,16

    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(tensor, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        #x = Dropout(0.3)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
    
    #x = GlobalAvgPool2D()(d)
    
    outputs = []
    for i in range(n_classes):
        print("\rclass ", i, end="")
        b = bn_rl_conv(d, f, 3)
        x = GlobalAvgPool2D()(b)
        output = Dense(1, activation=finalAct)(x)
        outputs.append(output)

    outputs = Concatenate()(outputs)
    #output = Dense(n_classes, activation=finalAct)(x)
    
    model = Model(input, outputs)
    
    return model

def densenet_ml4(img_shape, n_classes, finalAct='softmax', f = 32):
    repetitions = 6,12,24#,16

    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(x, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        #x = Dropout(0.3)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
    
    #x = GlobalAvgPool2D()(d)
    
    outputs = []
    for i in range(n_classes):
        ### Branches
        """
        x = Conv2D(32, 3, strides=2, padding='same')(d)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(32, 3, strides=1, padding='same')(x)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(32, 3, strides=1, padding='same')(x)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(32, 3, strides=1, padding='same')(x)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(32, 3, strides=1, padding='same')(x)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        """

        x = bn_rl_conv(d, f*4)
        x = bn_rl_conv(x, f, 3)
        x = bn_rl_conv(x, f*4)
        x = bn_rl_conv(x, f, 3)
        x = bn_rl_conv(x, f*4)
        x = bn_rl_conv(x, f, 3)
        #x = Dropout(0.2)(x)
        x = bn_rl_conv(x, f*4)
        x = bn_rl_conv(x, f, 3)
        x = bn_rl_conv(x, f*4)
        x = bn_rl_conv(x, f, 3)
        x = bn_rl_conv(x, f*4)
        x = bn_rl_conv(x, f, 3)
        x = bn_rl_conv(x, f*4)
        x = bn_rl_conv(x, f, 3)


        x = GlobalAvgPool2D()(x)

        output = Dense(1, activation=finalAct)(x)
        outputs.append(output)

    outputs = Concatenate()(outputs)
    #output = Dense(n_classes, activation=finalAct)(x)
    
    model = Model(input, outputs)
    
    return model

# CNN
def cnn(input_shape, num_classes, finalAct="softmax"):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=finalAct))
    
    return model

# AlexNet
def alexnet(input_shape, num_classes, finalAct="softmax"):
    model = Sequential()
    
    # 1st Convolutional Layer

    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    model.add(Dense(num_classes,activation=finalAct))

    return model

# VGG19
def vgg19(input_shape, num_classes, finalAct="softmax"):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation=finalAct))
    
    return model

def vgg(input_shape, num_classes, finalAct="softmax"):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation=finalAct))

    return model


# input_shape=(224,224,3)
# m = densenet(input_shape, 5)