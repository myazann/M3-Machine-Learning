from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras import optimizers 
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd

train_data_dir='datasets/MIT_small_train_1/train'
val_data_dir='datasets/MIT_small_train_1/test'
test_data_dir='datasets/MIT_small_train_2/test'

val_len = 0
for elem in os.listdir(val_data_dir):
 val_len += len(os.listdir(os.path.join(val_data_dir, elem)))

img_width = 256
img_height=256
batch_size=32
number_of_epoch=50
validation_samples=val_len

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
	
model = load_model("Last_Layer_Trained_Model.h5")


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
		
"""
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    decay_rate=0.25) 
optimizer = optimizers.Adam(learning_rate=lr_schedule)
"""

#csv_logger = CSVLogger('training.log')

model = Model(inputs=base_model.input, outputs=x)
for layer in model.layers:
    layer.trainable = True
	
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history=model.fit_generator(train_generator,
      steps_per_epoch=(int(400//batch_size)+1),
      epochs=50,
      validation_data=validation_generator,
      validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tensorboard_callback])

result = model.evaluate_generator(test_generator)

model.save("Finetuned_Model.h5")
