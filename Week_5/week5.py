from numpy import number
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Reshape, BatchNormalization
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras import optimizers 
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
"""
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.gpu_device_name()

train_data_dir='datasets/MIT_small_train_1/train'
val_data_dir='datasets/MIT_small_train_1/test'
test_data_dir='datasets/MIT_small_train_2/test'

val_len = 0
for elem in os.listdir(val_data_dir):
 val_len += len(os.listdir(os.path.join(val_data_dir, elem)))

img_width = 128
img_height = 128
batch_size = 32
number_of_epoch = 20
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

# create model
model = Sequential()
model.add(Reshape((img_width*img_height*3,), input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())

# model.add(Dense(units=2048, activation='relu'))
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True),
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
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

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
      steps_per_epoch=(int(100//batch_size)+1),
      epochs=number_of_epoch,
      validation_data=validation_generator,
      validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tensorboard_callback])

result = model.evaluate(test_generator)

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