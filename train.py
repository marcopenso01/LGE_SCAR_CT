import numpy as np
import math
from model import get_model
from matplotlib import pyplot as plt
import cv2
import os
import h5py
import tensorflow as 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

directory = 'F:\CT-tesi\data'
data = h5py.File(os.path.join(directory, 'tac_fold0.hdf5'), 'r')
train_images = data['segs_tr'][()]
train_labels = data['out_tr'][()]
val_images = data['segs_val'][()]
val_labels = data['out_val'][()]
test_images = data['paz_test'][()]
test_labels = data['out_test'][()]
test_patient = data['out_test'][()]
data.close()
print('training data', len(train_images), train_images[0].shape)
print('validation data', len(val_images), val_images[0].shape)

train_datagen = ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.1,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

test_datagen = ImageDataGenerator()

batch_size = 8
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
valid_generator = train_datagen.flow(val_images, val_labels, batch_size=batch_size)
test_generator = test_datagen.flow(test_images, batch_size=1)

model = get_model()
print('training started...')
history = model.fit(train_generator, epochs = 100, validation_data = valid_generator, verbose = 1, callbacks=[tensorboard])
print('training done...')

