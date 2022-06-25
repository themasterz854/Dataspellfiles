# first neural network with keras tutorial
from datetime import datetime

import keras.models
from keras import callbacks
from keras.layers import Dense
from keras.models import Sequential
from numpy import loadtxt

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#model.load_weights("./1.ckpt")

# compile the keras model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model = keras.models.load_model("x")
#checkpoint_path = "./1.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
'''cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)'''
logdir="./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
# Train the model with the new callback
model.fit(X,
          y,
          epochs=150, batch_size=500,callbacks=[tensorboard_callback])  # Pass callback to training'''
model.save("x")
# fit the keras model on the dataset
# model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
