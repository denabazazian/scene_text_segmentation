import cv2
import numpy as np
from PIL import Image
from numpy import *
import glob


gt_img_dir = '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/GT_color/'
#gt_img_dir = '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/resized_256/GT_color/'

#gt_lime_dir = '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/GT_color_LimeBG/'
gt_lime_dir = '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/resized_256/GT_color_limeBG/'

img_files = sorted(glob.glob(gt_img_dir+'*.png'))

#for img in img_files:
for img_idx in range(0,len(img_files)):
#for img_idx in range(0,5):

    img_name = img_files[img_idx].split('/')[-1].split('.')[0]
    print("image is {}".format(img_name))

    #image = Image.open(img)
    image = array(Image.open(img_files[img_idx]))

    ### convert black pixels to lime
    image[np.where((image==[0,0,0]).all(axis=2))] = [0,255,0]

    #image_lime = Image.fromarray(converted_image)
    image_lime = Image.fromarray(image)
    image_lime = image_lime.resize((256,256), Image.ANTIALIAS)
    image_lime.save(gt_lime_dir+img_name+'.png')
    print("done!")





