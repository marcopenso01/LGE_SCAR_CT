import numpy as np
import math
from model import get_model
from matplotlib import pyplot as plt
import cv2
import os
import h5py
import tensorflow as 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


directory = 'F:\CT-tesi\data'
output_folder = os.path.join(directory, 'fold0')
if not os.path.exists(output_folder):
    makefolder(output_folder)

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

callbacks=[EarlyStopping(patience=15,verbose=1),\
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001,verbose=1),\
                ModelCheckpoint('modelName.h5'.format(epochs),verbose=1, save_best_only=True,\
                                save_weights_only=False)

model = get_model()
print('training started...')
history = model.fit(train_generator, epochs = 200, validation_data = valid_generator, verbose = 1, callbacks=callbacks)
print('Model correctly trained and saved')  

plt.figure(figsize=(8, 8))
plt.grid(False)
plt.title("Learning curve LOSS", fontsize=25)
plt.plot(results.history["loss"], label="Loss")
plt.plot(results.history["val_loss"], label="Validation loss")
p=np.argmin(results.history["val_loss"])
plt.plot( p, results.history["val_loss"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend();
plt.savefig(output_folder+'Loss')

plt.figure(figsize=(8, 8))
plt.grid(False)
plt.title("Learning curve ACCURACY", fontsize=25)
plt.plot(results.history["accuracy"], label="Accuracy")
plt.plot(results.history["val_accuracy"], label="Validation Accuracy")
plt.plot( p, results.history["val_accuracy"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.legend();
plt.savefig(output_folder+'Accuracy')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 30)
print('Testing')
print('Loading saved weights...')
