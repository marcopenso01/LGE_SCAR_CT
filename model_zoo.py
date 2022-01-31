import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras import regularizers

initializer = tf.keras.initializers.GlorotNormal()
import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import Xception


def get_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=initializer, strides=(2,2),
                     input_shape=(85, 85, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

    return model

def residual_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer=initializer)(
            layer_in)
    # conv1
    conv1 = Conv2D(n_filters, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(layer_in)
    # conv2
    conv2 = Conv2D(n_filters, (3, 3), padding='same', activation='linear', kernel_initializer=initializer)(conv1)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out

def get_model2():

    input_tensor_shape = Input(shape=(85, 85, 1))
    x = residual_module(input_tensor_shape, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = residual_module(x, 48)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(96, (3, 3), padding='same', activation='relu', kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor_shape, output, name='cnn')
    return model


def VGG16_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])

    base_model = VGG16(input_tensor=images,
                       include_top=False,
                       weights='imagenet')
    #base_model.trainable = False
    #for layer in base_model.layers:
    #    print(layer)

    for layer in base_model.layers[:-2]:
        layer.trainable = False
    for layer in base_model.layers[-2:]:
        layer.trainable = True
        print(layer)

    '''
    model2 = Model(base_model.input, base_model.layers[4].output)
    for layer in model2.layers:
        print(layer)
    '''
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='VGG16')

    return model


def ResNet50V2_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])

    # include_top: whether to include the fully-connected layer at the top of the network.
    base_model = ResNet50V2(input_tensor=images,
                            include_top=False,
                            weights='imagenet')
    base_model.trainable = False
    '''
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True
        print(layer)
    '''
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='ResNet50V2')

    return model


def InceptionResNetV2_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 75.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])

    # include_top: whether to include the fully-connected layer at the top of the network.
    base_model = InceptionResNetV2(input_tensor=images,
                                   include_top=False,
                                   weights='imagenet')
    #base_model.trainable = False
    for layer in base_model.layers[:-3]:
        layer.trainable = False
    for layer in base_model.layers[-3:]:
        layer.trainable = True
        print(layer)

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='InceptionResNetV2')

    return model


def Xception_model():
    # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.
    input_tensor_shape = Input(shape=(85, 85, 1))
    images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])

    base_model = Xception(input_tensor=images,
                          include_top=False,
                          weights='imagenet')
    base_model.trainable = False
    #for layer in base_model.layers:
    #    print(layer)
    '''
    for layer in base_model.layers[:-2]:
        layer.trainable = False
    for layer in base_model.layers[-2:]:
        layer.trainable = True
        print(layer)
    '''
    '''
    model2 = Model(base_model.input, base_model.layers[4].output)
    for layer in model2.layers:
        print(layer)
    '''
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, output, name='Xception')

    return model
