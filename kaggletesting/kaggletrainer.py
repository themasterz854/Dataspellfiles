from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf

resnet_model = Sequential()
img_height, img_width = 180, 180
batch_size = 64
train_dir = "./trainimages/"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(180, 180, 3),
                                                  pooling='avg',
                                                  weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(2, activation="softmax"))

resnet_model.summary()

# resnet_model = tf.keras.models.load_model("kagglemodel")
resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)

print(resnet_model.save("kagglemodel"))
