import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Model, Input
initializer = tf.keras.initializers.GlorotNormal()
import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2

def get_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', padding="same", kernel_initializer=initializer, input_shape=(85,85,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2)) ## 
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.3)) ##
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer=initializer))

    return model


def VGG16_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
    
    base_model = VGG16(input_tensor = images,
                       include_top = False,
                       weights = 'imagenet')
    base_model.trainable = False
    '''
    for layer in base_model.layers[:-2]:
        layer.trainable = False
    for layer in base_model.layers[-2:]:
        layer.trainable = True
    '''
    
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu',kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output  = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='VGG16')
    
    return model


def ResNet50V2_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
    
    #include_top: whether to include the fully-connected layer at the top of the network. 
    base_model = ResNet50V2(input_tensor = images,
                            include_top = False,
                            weights = 'imagenet')
    base_model.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu',kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output  = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='ResNet50V2')
    
    return model


def InceptionResNetV2_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 75.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
    
    #include_top: whether to include the fully-connected layer at the top of the network. 
    base_model = InceptionResNetV2(input_tensor = images,
                                   include_top = False,
                                   weights = 'imagenet')
    base_model.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu',kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output  = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='InceptionResNetV2')
    
    return model
