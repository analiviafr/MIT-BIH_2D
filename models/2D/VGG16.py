from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16

def vgg16_model(input_shape, n_classes):
  input_tensor = Input(shape=input_shape)
  vgg16 = VGG16(include_top=False,
                  weights='imagenet',
                  input_tensor=input_tensor)

  top = Sequential()
  top.add(Flatten(input_shape=vgg16.output_shape[1:]))
  top.add(Dense(256, activation='relu'))
  top.add(Dropout(0.5))
  top.add(Dense(n_classes, activation='softmax'))
  
  model = Model(inputs=vgg16.input, outputs=top(vgg16.output))

  return model

if __name__ == '__main__':
  model = vgg16_model(input_shape=(256, 256, 3), n_classes=5)
  model.summary()
