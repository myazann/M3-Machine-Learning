from numpy import number
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras import optimizers, regularizers
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd


train_data_dir='datasets/MIT_large_train/train'
val_data_dir='datasets/MIT_small_train_1/test'
test_data_dir='datasets/MIT_large_train/test'

val_len = 0
for elem in os.listdir(val_data_dir):
 val_len += len(os.listdir(os.path.join(val_data_dir, elem)))

img_width = 64
img_height = 64
batch_size = 32
number_of_epoch = 70
validation_samples = val_len

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

"""
Flatten
Dense layers to help learn
BatchNormalization (speeds up learning)
Dropout (to prevent overfitting, they turn down some nodes)
Conv2D (applies kernel on the image)
MaxPooling2d (downsides the spatial dimensions)
"""

# create model
model = Sequential()
# model.add(Reshape((img_width,img_height,3), input_shape=(img_width, img_height, 3)))
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Dense(units=2048, activation='relu')) # , kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(units=2048, activation='relu')) # , kernel_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=1024, activation='relu'))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(units=512, activation='relu')) # , kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(units=512, activation='relu')) # , kernel_regularizer=regularizers.l1(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), activation="relu"))
# model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
# learning_rate=0.005, momentum=0.9, nesterov=True
print(model.summary())
plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

# exit()

# get data
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1/256)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# train the model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history=model.fit(train_generator,
      steps_per_epoch=(int(1800//batch_size)+1),
      epochs=number_of_epoch,
      validation_data=validation_generator,
      validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tensorboard_callback])

result = model.evaluate(test_generator)

model.save("week5.h5")

# plot results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.jpg')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.jpg')
