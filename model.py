import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
initializer = tensorflow.keras.initializers.GlorotNormal()
import os

def get_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', kernel_initializer=initializer, input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2)) ## 
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.3)) ##
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer=initializer))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print('model prepared...')
    return model
