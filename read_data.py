"""
@author: Marco Penso
"""
import scipy
import scipy.io
import os
import numpy as np
import logging
import h5py
from skimage import transform
import pydicom
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def setDicomWinWidthWinCenter(vol_data, winwidth, wincenter):
    vol_temp = np.copy(vol_data)
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    
    vol_temp = ((vol_temp[:]-min)*dFactor).astype('int16')

    min_index = vol_temp < 0
    vol_temp[min_index] = 0
    max_index = vol_temp > 255
    vol_temp[max_index] = 255

    return vol_temp


input_folder = r'F:\CT-tesi\Segmentation\1'
output_file = output_file = os.path.join(input_folder, 'pre_proc.hdf5')
hdf5_file = h5py.File(output_file, "w")

pat_addrs = {}
for ii in ['SEG', 'BAS', 'ART']:
  paths = [input_folder]
  pat_addrs[ii] = []
  while paths:
      with os.scandir(paths.pop()) as entries:
          for entry in entries:  # loop through the folder
              if entry.name.find(ii) != -1:
                  pat_addrs[ii].append(entry.path)
              elif entry.is_dir():  #if it is a subfolder
                  # Add to paths stack to get to it eventually
                  paths.append(entry.path)
                  
mat = scipy.io.loadmat(pat_addrs['BAS'][0])
vol_bas = mat['BAS1']
mat = scipy.io.loadmat(pat_addrs['SEG'][0])
vol_seg = mat['SEG1']
mat = scipy.io.loadmat(pat_addrs['ART'][0])
vol_art = mat['ART1']

vol_bas = vol_bas -1024
vol_art = vol_art -1024

vol_bas_transp = vol_bas.transpose([2,0,1])
vol_seg_transp = vol_seg.transpose([2,0,1])
vol_art_transp = vol_art.transpose([2,0,1])

vol_bas_flip = flip_axis(vol_bas_transp,1)
vol_seg_flip = flip_axis(vol_seg_transp,1)
vol_art_flip = flip_axis(vol_art_transp,1)

vol_bas_win = setDicomWinWidthWinCenter(vol_bas_flip, 300, 150)

for i in range(0, vol_bas_win.shape[0], 50):
  fig = plt.figure()
  plt.title(i)
  plt.imshow(vol_bas_win[i,...])
  plt.show()

#for i in range(vol_seg_flip.shape[0]):
#    if vol_seg_flip[i].max() > 0:
#        print(i)

first_sl = 395
last_sl = 565
vol_bas_flip = vol_bas_flip[395:565,...]
vol_seg_flip = vol_seg_flip[395:565,...]
vol_art_flip = vol_art_flip[395:565,...]
vol_bas_win = vol_bas_win[395:565,...]

X = []
Y = []
a = vol_bas_win[int(vol_bas_win.shape[0]/100*10),...]
a = a[...,np.newaxis]
b = np.concatenate((a,a,a,), axis=-1)
cv2.imshow("image", b.astype('uint8'))
cv2.namedWindow('image')
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)

a = vol_bas_win[int(vol_bas_win.shape[0]/100*90),...]
a = a[...,np.newaxis]
b = np.concatenate((a,a,a,), axis=-1)
cv2.imshow("image", b.astype('uint8'))
cv2.namedWindow('image')
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)

x = abs(int((X[0]+X[1])/2))
y = abs(int((Y[0]+Y[1])/2))

vol_bas_flip = vol_bas_flip[:,X[0]-110:X[0]+110, Y[0]-110:Y[0]+110]
vol_seg_flip = vol_seg_flip[:,X[0]-110:X[0]+110, Y[0]-110:Y[0]+110]
vol_art_flip = vol_art_flip[:,X[0]-110:X[0]+110, Y[0]-110:Y[0]+110]
vol_bas_win = vol_bas_win[:,X[0]-110:X[0]+110, Y[0]-110:Y[0]+110]

print(vol_bas_flip.shape, vol_bas_flip.dtype, vol_bas_flip.min(), vol_bas_flip.max())
print(vol_seg_flip.shape, vol_seg_flip.dtype, vol_seg_flip.min(), vol_seg_flip.max())
print(vol_art_flip.shape, vol_art_flip.dtype, vol_art_flip.min(), vol_art_flip.max())
print(vol_bas_win.shape, vol_bas_win.dtype, vol_bas_win.min(), vol_bas_win.max())

for i in range(0, vol_bas_win.shape[0], 10):
  fig = plt.figure()
  plt.title(i)
  plt.imshow(vol_bas_win[i,...])
  plt.show()

num_slices = vol_bas_flip.shape[0]
size = vol_bas_flip.shape[1:3]

hdf5_file.create_dataset('LGE', [num_slices] + list(size), dtype=np.int16)
hdf5_file.create_dataset('SEG', [num_slices] + list(size), dtype=np.uint8)
hdf5_file.create_dataset('ART', [num_slices] + list(size), dtype=np.int16)
hdf5_file.create_dataset('LGEwin', [num_slices] + list(size), dtype=np.uint8)

hdf5_file['LGE'][:] = vol_bas_flip[None]
hdf5_file['SEG'][:] = vol_seg_flip[None]
hdf5_file['ART'][:] = vol_art_flip[None]
hdf5_file['LGEwin'][:] = vol_bas_win[None]

hdf5_file.close()
  
