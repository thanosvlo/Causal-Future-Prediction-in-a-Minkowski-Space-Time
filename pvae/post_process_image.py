import os
import glob
import numpy as np
from PIL import Image
import cv2


base_dir = '/Users/tvlo/PycharmProjects/CrystalCones/Manifolds/imgdump/'

dirs = glob.glob(os.path.join(base_dir,'*/selected/'))

width = 36
height = 32



for dir in dirs:
    out_path = dir.replace('selected', 'selected_proc')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_paths = glob.glob(os.path.join(dir, '*.png'))
    for jj, image in enumerate(image_paths):
        print(jj)
        img = cv2.imread(image)
        imgheight, imgwidth, channels = img.shape
        number_of_frames = np.floor(imgwidth/width)
        bsname = os.path.basename(image).split('.')[0]
        l = 0

        for j in range(0, imgwidth, width):
            # if j+width <= imgwidth:
            img_proc = img[:, j:j+width, :]
            img_proc = cv2.resize(img_proc, (height,width))

            indx = int(np.abs(l - number_of_frames))
            l += 1

            cv2.imwrite(os.path.join(out_path,bsname+'_proc_{}.png'.format(indx)),img_proc)


