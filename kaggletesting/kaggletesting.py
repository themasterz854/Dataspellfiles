import os
import gc
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from matplotlib import patches

get_ipython().run_line_magic('matplotlib', 'inline')



train = pd.read_csv("train_anno_kaggle.csv")
train['filepath'].nunique()
train['class_name'].value_counts()
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

classes = ['phone', 'no_phone']
