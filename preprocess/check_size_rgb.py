import cv2
import numpy as np
from PIL import Image
from numpy import *
import glob

gt_img_dir = '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/GT_color/'
save_path = 'pat/to/save'

count_wrong_size = 0 
img_files = sorted(glob.glob(gt_img_dir+'*.png'))

for img_idx in range(0,len(img_files)):

	img_name = img_files[img_idx].split('/')[-1].split('.')[0]
    print("image is {}".format(img_name))

    #image = Image.open(img)
    image = Image.open(img_files[img_idx])

    img_size = image.size

    if img_size[0] != 250 and img_size[1] != 250: 
    	count_wrong_size +=1 
    	image_resized = image.resize((256,256), Image.ANTIALIAS)
    	image_resized.save(save_path+img_name+'.png')

print(count_wrong_size)
