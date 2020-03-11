from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Conv2D, MaxPooling2D, ZeroPadding2D)

def vgg19_model(weights=False, summary=False):
    vgg19 = Sequential()

    vgg19.add(ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
    vgg19.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
    vgg19.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
    vgg19.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, 3, 3, activation='relu', name='conv3_4'))
    vgg19.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv4_4'))
    vgg19.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, 3, 3, activation='relu', name='conv5_4'))
    vgg19.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    vgg19.add(Flatten())
    vgg19.add(Dense(4096, activation='relu', name='dense_1'))
    vgg19.add(Dropout(0.5))
    vgg19.add(Dense(4096, activation='relu', name='dense_2'))
    vgg19.add(Dropout(0.5))
    vgg19.add(Dense(1000, name='dense_3'))
    vgg19.add(Activation("softmax"))

    return vgg19

if __name__ == '__main__':
    model = vgg19_model()
    model.summary()