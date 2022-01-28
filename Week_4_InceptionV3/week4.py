import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import keras


# train_data_dir='/home/mcv/datasets/MIT_split/train'
# val_data_dir='/home/mcv/datasets/MIT_split/test'
# test_data_dir='/home/mcv/datasets/MIT_split/test'
train_data_dir='../MIT_split/train'
val_data_dir='../MIT_split/test'
test_data_dir='../MIT_split/test'

"""
img_width = 299
img_height=299
batch_size=32
number_of_epoch=100
validation_samples=807
"""

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
# create the base pre-trained model
# base_model = VGG16(weights='imagenet')
# plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)
base_model = InceptionV3(weights='imagenet')
plot_model(base_model, to_file='modelInceptionV3a.png', show_shapes=True, show_layer_names=True)

x = base_model.layers[-2].output
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(inputs=base_model.input, outputs=x)
# plot_model(model, to_file='modelVGG16b.png', show_shapes=True, show_layer_names=True)
plot_model(model, to_file='modelInceptionV3b.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = False
    
    
model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
    print(layer.name, layer.trainable)

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	preprocessing_function=preprocess_input,
    rotation_range=0.5,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1/255)

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

history=model.fit_generator(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        epochs=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[])


result = model.evaluate_generator(test_generator)
print( result)
"""

nb_classes = 8  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
nb_train_samples = 1881  # Total number of train samples. NOT including augmented images | 1881, 2000
nb_validation_samples = 807  # Total number of train samples. NOT including augmented images. | 807, 800
model_path = '.'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)

# # Top Model Block    
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)

predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
print(model.summary())
# bottleneck featuring 
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all layers of the based model that is already pre-trained.
for layer in base_model.layers:
    layer.trainable = False

# Data augmentation 
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range=transformation_ratio,
                                    shear_range=transformation_ratio,
                                    zoom_range=transformation_ratio,
                                    cval=transformation_ratio,
                                    horizontal_flip=True,
                                    vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(val_data_dir,
                                                            target_size=[img_width, img_height],
                                                            batch_size=batch_size,
                                                            class_mode='categorical')
# Model compiling
model.compile(optimizer=tf.keras.optimizers.Adam(learn_rate),
                loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                metrics=['accuracy'])

# save weights of best training epoch

top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
callbacks_list = [
    ModelCheckpoint(top_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_accuracy', patience=5, verbose=0),
    keras.callbacks.TensorBoard(log_dir='tensorboard/inception-v3-train-top-layer', histogram_freq=0, write_graph=False, write_images=False)
]

# Train Simple CNN
history = model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epoch // 5,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=callbacks_list)

# list all data in history

if True:
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy_inception_V3.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss_inception_v3.jpg')
