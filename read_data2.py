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

X = []
Y = []

drawing=False # true if mouse is pressed
mode=True

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
  
def imfill(img, dim):
    img = img[:,:,0]
    img = cv2.resize(img, (dim, dim))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

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


def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped

def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped


input_folder = r'F:\CT-tesi\Segmentation'
patient = 47

input_folder = os.path.join(input_folder, str(patient))
               
mat = scipy.io.loadmat(os.path.join(input_folder, 'ART1.mat'))
vol_art = mat['ART1']
mat = scipy.io.loadmat(os.path.join(input_folder, 'SEG1.mat'))
vol_seg = mat['SEG1']

vol_art = vol_art -1024

vol_art = flip_axis(vol_art.transpose([2,0,1]),1)
vol_seg = flip_axis(vol_seg.transpose([2,0,1]),1)

art_imgs = []
seg_imgs = []

for i in range(len(vol_seg)):
    if vol_seg[i,...].max() != 0:
        print(i, vol_seg[i,...].max())
        art_imgs.append(vol_art[i,...])
        seg_imgs.append(vol_seg[i,...])

art_imgs = np.asarray(art_imgs)
seg_imgs = np.asarray(seg_imgs)

# select center LV
print('select center LV')
art_crop = []
seg_crop = []
X = []
Y = []
img = art_imgs[len(art_imgs)//2,...].copy()
img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("image", img.astype('uint8'))
cv2.namedWindow('image')
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(len(art_imgs)):
    art_crop.append(crop_or_pad_slice_to_size_specific_point(art_imgs[i,...], 250, 250, X[0], Y[0]))
    seg_crop.append(crop_or_pad_slice_to_size_specific_point(seg_imgs[i,...], 250, 250, X[0], Y[0]))

art_crop = np.asarray(art_crop).astype('float32')
seg_crop = np.asarray(seg_crop).astype('uint8')

'''
for i in range(len(art_crop)):
    img = art_crop[i,...].copy()
    obj = cv2.normalize(src=seg_crop[i,...], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    clahe = cv2.createCLAHE(clipLimit = 1.8)
    img = clahe.apply(img)
    plt.figure()
    plt.imshow(img)
'''
# segmentation Myo
myo = []                         # myo 
mask_myo = []                    # maschera binaria myo 
segments = []                    # segmento myo 
mask_segments = []               # maschera binaria segmento myo
seg_cropped = []                 # segmento myo croppato
mask_seg_cropped = []            # maschera binaria segmento myo croppato
AHA = []                         # 1 se segmento definito secondo AHA model, 0 altrimenti
scar = []                        # maschera binaria scar
mask_segment_scar = []           # maschera segmento myo + scar
scar_cropped = []                # maschera segmento myo + scar, croppato
scar_area = []                   # percentuale area scar in segmento myo 


tit=['epicardium', 'endocardium','scar']
for i in range(3, len(art_crop)-3):
    print("{}/{}".format(i, len(art_crop)))
    print('---select the point where the RV wall joins the LV')
    X = []
    Y = []
    slice_crop = cv2.normalize(src=art_crop[i,...], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("image", slice_crop)
    cv2.namedWindow('image')
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    print('---Segmenting myocardium:')
    for ii in range(3):
        img = art_crop[i,...].copy()
        dim = img.shape[0]
        obj = cv2.normalize(src=seg_crop[i,...], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        clahe = cv2.createCLAHE(clipLimit = 1.5)
        img = clahe.apply(img)
        img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
        image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        cv2.namedWindow(tit[ii])
        cv2.setMouseCallback(tit[ii],paint_draw)
        while(1):
            cv2.imshow(tit[ii],img)
            k=cv2.waitKey(1)& 0xFF
            if k==27: #Escape KEY
                if ii==0:   
                    im_out1 = imfill(image_binary, dim)
                    im_out1[im_out1>0]=255                    
                elif ii==1:                                         
                    im_out2 = imfill(image_binary, dim)
                    im_out2[im_out2>0]=255
                    
                    contours, _ = cv2.findContours(im_out2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                    ref = np.zeros_like(im_out2)
                    cv2.drawContours(ref, contours, 0, 255, 1);
                    M = cv2.moments(ref)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                elif ii==2:  
                    scar_out1 = imfill(image_binary, dim)
                    scar_out1[scar_out1>0]=255
                    
                    
                break
        cv2.destroyAllWindows()  

    im_out1[im_out1>0]=1
    im_out2[im_out2>0]=1
    scar_out1[scar_out1>0]=1
    mask = im_out1 - im_out2
    plt.figure()
    plt.imshow(mask)
    plt.title(i)
        
    # AHA model
    N = 6
    nn = 85
    phi = round(np.rad2deg(math.atan2(cY - cY, X[0] - cX) - math.atan2(Y[0] - cY, X[0] - cX)))
    contours, _ = cv2.findContours(im_out1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    ref = np.zeros_like(im_out1)
    cv2.drawContours(ref, contours, 0, 255, 1);
    for val in range(4):
        if val!=0:
            phi = random.randint(1,355)
        for n in range(N):
            tmp = np.zeros_like(art_crop[i,...])
            for xx in range(2):
                theta = (n+xx)*(360/N)-90-phi
                theta *= np.pi/180.0
                cv2.line(tmp, (cX, cY),
                         (int(cX+np.cos(theta)*ref.shape[0]),
                          int(cY-np.sin(theta)*ref.shape[0])), 255, 1);
            tmp = tmp[..., np.newaxis].astype(np.uint8)
            tmp = imfill(tmp, tmp.shape[0])
            if tmp.min() == 255:
                tmp = np.zeros_like(art_crop[i,...])
                for xx in range(2):
                    theta = (n+xx)*(360/N)-90-(phi+1)
                    theta *= np.pi/180.0
                    cv2.line(tmp, (cX, cY),
                             (int(cX+np.cos(theta)*ref.shape[0]),
                              int(cY-np.sin(theta)*ref.shape[0])), 255, 1);
                tmp = tmp[..., np.newaxis].astype(np.uint8)
                tmp = imfill(tmp, tmp.shape[0])
            tmp2 = np.invert(tmp)
            if np.count_nonzero(tmp) > np.count_nonzero(tmp2):
                tmp = tmp2
            tmp[tmp>0]=1
            # binary mask of the segment
            out = tmp & mask        
            mask_segments.append(out.astype(np.uint8))
            # scar
            scar.append((seg_crop[i,...].astype(np.uint8)))
            # mask segment with scar
            a = out.astype(np.uint8)
            b = scar_out1.astype(np.uint8)
            mask_segment_scar.append((a+b)*a)
            #crop mask segment
            contours, hier = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            top_left_x = 1000
            top_left_y = 1000
            bottom_right_x = 0
            bottom_right_y = 0
            for cntr in contours:
                x,y,w,h = cv2.boundingRect(cntr)
                if x < top_left_x:
                    top_left_x = x
                if y < top_left_y:
                    top_left_y= y
                if x+w-1 > bottom_right_x:
                    bottom_right_x = x+w-1
                if y+h-1 > bottom_right_y:
                    bottom_right_y = y+h-1        
            top_left = (top_left_x, top_left_y)
            bottom_right = (bottom_right_x, bottom_right_y)
            cx = int((top_left[1]+bottom_right[1])/2)   #row
            cy = int((top_left[0]+bottom_right[0])/2)   #column
            len_x = int(bottom_right[1]-top_left[1])
            len_y = int(bottom_right[0]-top_left[0])
            lenn = max(len_x, len_y)
            #print(lenn)
            out_crop = crop_or_pad_slice_to_size_specific_point(out,lenn,lenn,cx,cy)
            if lenn < nn:
                out_crop = crop_or_pad_slice_to_size(out_crop, nn, nn)
            elif lenn > nn:
                out_crop = cv2.resize(out_crop, (nn, nn), interpolation=cv2.INTER_NEAREST)
            mask_seg_cropped.append(out_crop.astype(np.uint8))
            # scar_cropped
            scarcrop = crop_or_pad_slice_to_size_specific_point((a+b)*a,lenn,lenn,cx,cy)
            if lenn < nn:
                scarcrop = crop_or_pad_slice_to_size(scarcrop, nn, nn)
            elif lenn > nn:
                scarcrop = cv2.resize(scarcrop, (nn, nn), interpolation=cv2.INTER_NEAREST)
            scarcrop = scarcrop.astype(np.uint8)
            scar_cropped.append(scarcrop)
            # % scar
            print(np.sum(scarcrop == 2), np.sum(scarcrop > 0))
            scar_area.append(int(np.sum(scarcrop == 2)*100/np.sum(scarcrop > 0)))
            #segment
            seg = art_crop[i,...]*out
            segments.append(seg.astype(np.float32))
            #crop segment
            segcrop = crop_or_pad_slice_to_size_specific_point(seg,lenn,lenn,cx,cy)
            if lenn < nn:
                segcrop = crop_or_pad_slice_to_size(segcrop, nn, nn)
            elif lenn > nn:
                segcrop = cv2.resize(segcrop, (nn, nn), interpolation=cv2.INTER_NEAREST)
            seg_cropped.append(segcrop.astype(np.float32))
            #mask myo
            mask_myo.append(mask.astype(np.uint8))
            #myo
            myo.append((art_crop[i,...]*mask).astype(np.float32))
            #
            if val==0:
                AHA.append(1)
            else:
                AHA.append(0)

print('---Segmentation correctly completed')
print('Saving data...')
output_file = os.path.join(input_folder, 'tesi_tac.hdf5')
hdf5_file = h5py.File(output_file, "w")
n1 = art_crop.shape[1]
n2 = seg_cropped[0].shape[0]
#dt = h5py.special_dtype(vlen=str)

hdf5_file.create_dataset('paz', (len(myo),), dtype=np.uint8)
hdf5_file.create_dataset('myo', [len(myo)] + [n1, n1], dtype=np.float32)
hdf5_file.create_dataset('mask_myo', [len(mask_myo)] + [n1, n1], dtype=np.uint8)
hdf5_file.create_dataset('segments', [len(segments)] + [n1, n1], dtype=np.float32)
hdf5_file.create_dataset('mask_segments', [len(mask_segments)] + [n1, n1], dtype=np.uint8)
hdf5_file.create_dataset('seg_cropped', [len(segments)] + [n2, n2], dtype=np.float32)
hdf5_file.create_dataset('mask_seg_cropped', [len(mask_segments)] + [n2, n2], dtype=np.uint8)
hdf5_file.create_dataset('AHA', (len(AHA),), dtype=np.uint8)
hdf5_file.create_dataset('scar', [len(scar)] + [n1, n1], dtype=np.uint8)
hdf5_file.create_dataset('mask_segment_scar', [len(mask_segment_scar)] + [n1, n1], dtype=np.uint8)
hdf5_file.create_dataset('scar_cropped', [len(scar_cropped)] + [n2, n2], dtype=np.uint8)
hdf5_file.create_dataset('scar_area', (len(scar_area),), dtype=np.uint8)


for i in range(len(myo)):
     hdf5_file['paz'][i, ...] = patient
     hdf5_file['myo'][i, ...] = myo[i]
     hdf5_file['mask_myo'][i, ...] = mask_myo[i]
     hdf5_file['segments'][i, ...] = segments[i]
     hdf5_file['mask_segments'][i, ...] = mask_segments[i]
     hdf5_file['seg_cropped'][i, ...] = seg_cropped[i]
     hdf5_file['mask_seg_cropped'][i, ...] = mask_seg_cropped[i]
     hdf5_file['AHA'][i, ...] = AHA[i]
     hdf5_file['scar'][i, ...] = scar[i]
     hdf5_file['mask_segment_scar'][i, ...] = mask_segment_scar[i]
     hdf5_file['scar_cropped'][i, ...] = scar_cropped[i]
     hdf5_file['scar_area'][i, ...] = scar_area[i]
     
# After loop:
hdf5_file.close()
    
    
#How to show contours Myo
#contours, hierarchy = cv2.findContours(mask_myo[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.normalize(src=art_crop[0,...], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#a = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#plt.imshow(a)
