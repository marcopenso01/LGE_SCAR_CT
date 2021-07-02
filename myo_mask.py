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


path = r'F:\CT-tesi\Segmentation\1\xxx.hdf5'
data = h5py.File(path, 'r+')

Min = 0
Max = 0
flag = True
for i in range(data['LGEwin'].shape[0]):
    if data['SEG'][i].max() > 0:
        if flag:
            Min = i
            flag = False
        else:
            Max = i

mask = []
tit=['epicardium', 'endocardium']
for i in range(Min, Max, 1):
    
    for ii in range(2):
        img = data['LGEwin'][i]
        img = np.array(img)
        
        scar = data['SEG'][i]
        max_index = scar > 0
        lge_scar = np.copy(img)
        lge_scar[max_index] = 255
        fig = plt.figure()
        plt.imshow(lge_scar, cmap='gray')
        plt.title('IMAGE WITH SCAR')
        
        scar = data['ART'][i]
        max_index = scar > 0
        art_scar = np.copy(img)
        art_scar[max_index] = 1500
        fig = plt.figure()
        plt.imshow(art_scar, cmap='gray')
        plt.title('IMAGE WITH SCAR')
        
        img = cv2.resize(lge_scar, (300, 300), interpolation = cv2.INTER_CUBIC)

        image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

        cv2.namedWindow(tit[ii])
        cv2.setMouseCallback(tit[ii],paint_draw)
        while(1):
            cv2.imshow(tit[ii],img)
            k=cv2.waitKey(1)& 0xFF
            if k==27: #Escape KEY
                if ii==0:
                    
                    im_out1 = imfill(image_binary, img.shape[0])
                    im_out1[im_out1>0]=255
                    print(im_out1.shape)
                    #fig = plt.figure()
                    #plt.imshow(im_out1)
                    #plt.show()
                    
                elif ii==1:
                                            
                    im_out2 = imfill(image_binary, img.shape[0])
                    im_out2[im_out2>0]=255
                    #fig = plt.figure()
                    #plt.imshow(im_out2)
                    #plt.show()
                break
        cv2.destroyAllWindows()
    myo_mask = im_out1 - im_out2
    myo_mask[myo_mask>0]=1
    mask.append(myo_mask)

num_slices = data['LGE'].shape[0]
size = data['LGE'].shape[1:3]
data.create_dataset('mask', [num_slices] + list(size), dtype=np.uint8)
