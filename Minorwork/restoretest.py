# first neural network with keras tutorial
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.load_weights("./1.ckpt")