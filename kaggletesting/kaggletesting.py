import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential

class_names = ["nophone", "Phone"]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

resnet_model = Sequential()
img_height, img_width = 180, 180
batch_size = 64
train_dir = "./trainimages/"


resnet_model = keras.models.load_model("kagglemodel")
resnet_model.summary()



image = cv2.imread("C:/Users/Zaid/Desktop/Datasets/Kaggle/test/img_8.jpg")
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)

pred = resnet_model.predict(image)
output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)

image = cv2.imread("C:/Users/Zaid/Desktop/Datasets/Kaggle/test/img_9.jpg")
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)

pred = resnet_model.predict(image)
output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)

image = cv2.imread("C:/Users/Zaid/Desktop/Datasets/Kaggle/test/img_84.jpg")
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)

pred = resnet_model.predict(image)
output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)
image = cv2.imread("C:/Users/Zaid/Desktop/Datasets/Kaggle/test/img_3.jpg")
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)

pred = resnet_model.predict(image)
output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)
