from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Conv2D, MaxPooling2D, ZeroPadding2D)

def vgg16_model(weights=False, summary=False):
    vgg16 = Sequential()
    vgg16.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    vgg16.add(Conv2D(64, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(64, 3, 3, activation='relu'))
    vgg16.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(128, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(128, 3, 3, activation='relu'))
    vgg16.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(256, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(256, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(256, 3, 3, activation='relu'))
    vgg16.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, 3, 3, activation='relu'))
    vgg16.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, 3, 3, activation='relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, 3, 3, activation='relu'))
    vgg16.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16.add(Flatten())
    vgg16.add(Dense(4096, activation='relu'))
    vgg16.add(Dropout(0.5))
    vgg16.add(Dense(4096, activation='relu'))
    vgg16.add(Dropout(0.5))
    vgg16.add(Dense(1000, activation='softmax'))

    return vgg16

if __name__ == '__main__':
    model = vgg16_model()
    model.summary()