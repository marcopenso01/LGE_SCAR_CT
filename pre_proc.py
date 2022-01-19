import scipy
import scipy.io
import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import math
import random
import pydicom

path = r'F:/CT-tesi/Pre-proc'

paz = []
segs = []
out = []

for paz in os.listdir(path):
    data = h5py.File(os.path.join(path, paz, 'tesi_tac.hdf5'), "r")

    #data.keys()
    n_file = len(data['myo'])
    
    random_indices = np.arange(n_file)
    np.random.shuffle(random_indices)
    count_1 = 0
    count_0 = 0
    
    for i in range(n_file):
        if data['scar_area'][i] > 10:
            segs.append(data['seg_cropped'][i])
            out.append(1)
            paz.append(data['paz'][i])
            count_1 += 1
    for i in range(n_file):
        if data['scar_area'][random_indices[i]] == 0:
            segs.append(data['seg_cropped'][random_indices[i]])
            out.append(0)
            paz.append(data['paz'][random_indices[i]])
            count_0 += 1
        if count_0 >= count_1:
            break
            
    data.close()
