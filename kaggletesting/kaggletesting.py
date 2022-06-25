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

fig = plt.figure()

# add axes to the image
ax = fig.add_axes([0, 0, 1, 1])

image = plt.imread('C:/Users/Zaid/Dataspellfiles/kaggletesting/train/c1/img_700.jpg')
plt.imshow(image)
# iterating over the image for different objects
directory_list = os.listdir()
print("Files and directories in  current working directory :")
print(directory_list)
for _, row in train[train.filepath == "C:/Users/Zaid/Dataspellfiles/kaggletesting/train/c1/img_700.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax

    width = xmax - xmin
    height = ymax - ymin

    # assign different color to different classes of objects
    edgecolor = 'g'
    ax.annotate('Phone', xy=(xmax, ymin))
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')
    ax.add_patch(rect)
plt.axis('off')
plt.savefig("foo.png")
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
