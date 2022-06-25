import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential

resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(180, 180, 3),
                                                  pooling='avg', classes=5,
                                                  weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))

resnet_model.summary()
