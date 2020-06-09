from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.vgg19 import VGG19

def vgg19_model(input_shape, num_genres, freezed_layers=5):
  input_tensor = Input(shape=input_shape)
  vgg16 = VGG19(include_top=False,
                  weights='imagenet',
                  input_tensor=input_tensor)

  top = Sequential()
  top.add(Flatten(input_shape=vgg16.output_shape[1:]))
  top.add(Dense(256, activation='relu'))
  top.add(Dropout(0.5))
  top.add(Dense(num_genres, activation='softmax'))
  
  model = Model(inputs=vgg16.input, outputs=top(vgg16.output))
  for layer in model.layers[:freezed_layers]:
    layer.trainable = False

  return model

if __name__ == '__main__':
  model = vgg19_model(input_shape=(256, 256, 3), num_genres=5)
  model.summary()
