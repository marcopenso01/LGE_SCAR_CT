"""
Created on Wed Jan 19 09:24:44 2022

@author: Marco Penso
"""

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

def standardize_image(image, mask):
    '''
    make image zero mean and unit standard deviation
    '''
    px = []
    coor = np.where(mask == 1)
    for ii in range(len(coor[0])):
        px.append(image[coor[0][ii],coor[1][ii]])
    px = np.asarray(px)
    img_o = np.float32(image.copy())
    m = np.mean(px)
    s = np.std(px)
    return np.divide((img_o - m), s)
    
    
path = r'F:/CT-tesi/Pre-proc'
out_path r'F:/CT-tesi/data'

n_paz = len(os.listdir(path))
ind_paz = np.arange(n_paz)
random.Random(4).shuffle(ind_paz)

# ciclo su fold (5-fold)
for b_i in range(5):
    
    print('-------------------------------------')
    print('fold: %s' % b_i)
    
    test_ind = ind_paz[b_i*10:b_i*10+10]
    tr = []
    train_ind = []
    val_ind = []
    for ii in range(n_paz):
        if ii < b_i*10 or ii >= b_i*10+10:
            tr.append(ind_paz[ii])
    
    rd_ind = np.arange(len(tr))
    random.Random(1).shuffle(rd_ind)
    for ii in range(len(rd_ind)):
        if ii<5:
            val_ind.append(tr[rd_ind[ii]])
        else:
            train_ind.append(tr[rd_ind[ii]])
    
    
    paz_test = []
    segs_test = []
    out_test = []
    paz_tr = []
    segs_tr = []
    out_tr = []
    paz_val = []
    segs_val = []
    out_val = []
    
    # test patients
    for zz in range(len(test_ind)):
        
        print('processing test patients: %s' % os.listdir(path)[test_ind[zz]])
        data = h5py.File(os.path.join(path, os.listdir(path)[test_ind[zz]], 'tesi_tac.hdf5'), "r")
        #data.keys()
        n_file = len(data['myo'])
        
        for i in range(n_file):
            if data['AHA'][i] == 1:
                if data['scar_area'][i] > 10:
                    img = standardize_image(data['seg_cropped'][i], data['mask_seg_cropped'][i])
                    segs_test.append(img)
                    out_test.append(1)
                    paz_test.append(data['paz'][i])
                elif data['scar_area'][i] == 0:
                    img = standardize_image(data['seg_cropped'][i], data['mask_seg_cropped'][i])
                    segs_test.append(img)
                    out_test.append(0)
                    paz_test.append(data['paz'][i])
                
        data.close()
    
    # train patients
    for zz in range(len(train_ind)):
        
        print('processing train patients: %s' % os.listdir(path)[train_ind[zz]])
        data = h5py.File(os.path.join(path, os.listdir(path)[train_ind[zz]], 'tesi_tac.hdf5'), "r")
        #data.keys()
        n_file = len(data['myo'])
        
        random_indices = np.arange(n_file)
        np.random.shuffle(random_indices)
        count_1 = 0
        count_0 = 0
        
        for i in range(n_file):
            if data['scar_area'][i] > 10:
                img = standardize_image(data['seg_cropped'][i], data['mask_seg_cropped'][i])
                segs_tr.append(img)
                out_tr.append(1)
                paz_tr.append(data['paz'][i])
                count_1 += 1
        for i in range(n_file):
            if data['scar_area'][random_indices[i]] == 0:
                img = standardize_image(data['seg_cropped'][random_indices[i]], data['mask_seg_cropped'][random_indices[i]])
                segs_tr.append(img)
                out_tr.append(0)
                paz_tr.append(data['paz'][random_indices[i]])
                count_0 += 1
            if count_0 >= count_1:
                break
                
        data.close()
        
    # val patients
    for zz in range(len(val_ind)):
        
        print('processing val patients: %s' % os.listdir(path)[val_ind[zz]])
        data = h5py.File(os.path.join(path, os.listdir(path)[val_ind[zz]], 'tesi_tac.hdf5'), "r")
        #data.keys()
        n_file = len(data['myo'])
        
        random_indices = np.arange(n_file)
        np.random.shuffle(random_indices)
        count_1 = 0
        count_0 = 0
        
        for i in range(n_file):
            if data['scar_area'][i] > 10:
                img = standardize_image(data['seg_cropped'][i], data['mask_seg_cropped'][i])
                segs_val.append(img)
                out_val.append(1)
                paz_val.append(data['paz'][i])
                count_1 += 1
        for i in range(n_file):
            if data['scar_area'][random_indices[i]] == 0:
                img = standardize_image(data['seg_cropped'][random_indices[i]], data['mask_seg_cropped'][random_indices[i]])
                segs_val.append(img)
                out_val.append(0)
                paz_val.append(data['paz'][random_indices[i]])
                count_0 += 1
            if count_0 >= count_1:
                break
                
        data.close()

    #create hdf5 file
    print('saving file...')
    hdf5_file = h5py.File(os.path.join(out_path, 'tac_fold'+str(b_i)+'.hdf5'), "w")
    
    hdf5_file.create_dataset('paz_test', (len(paz_test),), dtype=np.uint8)
    hdf5_file.create_dataset('segs_test', [len(segs_test)] + [85, 85], dtype=np.float32)
    hdf5_file.create_dataset('out_test', (len(out_test),), dtype=np.uint8)
    hdf5_file.create_dataset('paz_tr', (len(paz_tr),), dtype=np.uint8)
    hdf5_file.create_dataset('segs_tr', [len(segs_tr)] + [85, 85], dtype=np.float32)
    hdf5_file.create_dataset('out_tr', (len(out_tr),), dtype=np.uint8)
    hdf5_file.create_dataset('paz_val', (len(paz_val),), dtype=np.uint8)
    hdf5_file.create_dataset('segs_val', [len(segs_val)] + [85, 85], dtype=np.float32)
    hdf5_file.create_dataset('out_val', (len(out_val),), dtype=np.uint8)
    
    for i in range(len(paz_test)):
         hdf5_file['paz_test'][i, ...] = paz_test[i]
         hdf5_file['segs_test'][i, ...] = segs_test[i]
         hdf5_file['out_test'][i, ...] = out_test[i]
         
    for i in range(len(paz_tr)):
         hdf5_file['paz_tr'][i, ...] = paz_tr[i]
         hdf5_file['segs_tr'][i, ...] = segs_tr[i]
         hdf5_file['out_tr'][i, ...] = out_tr[i]

    for i in range(len(paz_val)):
         hdf5_file['paz_val'][i, ...] = paz_val[i]
         hdf5_file['segs_val'][i, ...] = segs_val[i]
         hdf5_file['out_val'][i, ...] = out_val[i]
    hdf5_file.close()
