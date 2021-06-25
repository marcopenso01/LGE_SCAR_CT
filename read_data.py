"""
@author: Marco Penso
"""
import os
import numpy as np
import logging
import h5py
import scipy.io
from skimage import transform

logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

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
  
