import cv2
import numpy as np
from PIL import Image
from numpy import *
import glob

img_dir = "/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/inputImg/"
gt_dir = "/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/GT_color/"

#gt_lime_dir = '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/resized_256/GT_color_limeBG/'
image_resized_dir = "/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/resized_512/inputImg/"
gt_resized_dir = "/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/resized_512/gt_color/"


img_files = sorted(glob.glob(img_dir+'*.jpg')) #'*.jpg')) #'*.png'))
gt_files = sorted(glob.glob(gt_dir+'*.png'))

#for img in img_files:
for img_idx in range(0,len(img_files)):
#for img_idx in range(0,5):

    img_name = img_files[img_idx].split('/')[-1].split('.')[0]
    print("image is {}".format(img_name))

    gt_name = gt_files[img_idx].split('/')[-1].split('.')[0]
    print("GT is {}".format(gt_name))

    #image = Image.open(img)
    #image = array(Image.open(img_files[img_idx]))
    image = Image.open(img_files[img_idx])
    gt = Image.open(gt_files[img_idx])

    ### convert black pixels to lime
    #image[np.where((image==[0,0,0]).all(axis=2))] = [0,255,0]

    #image_lime = Image.fromarray(converted_image)
    #image_lime = Image.fromarray(image)
    #image_lime = image_lime.resize((256,256), Image.ANTIALIAS)
    #image_resized = image.resize((256,256), Image.ANTIALIAS)
    #image_resized = image.resize((64,64), Image.ANTIALIAS)

    image_resized = image.resize((512,512), Image.ANTIALIAS)
    gt_resized = gt.resize((512,512), Image.ANTIALIAS)

    #image_lime.save(gt_lime_dir+img_name+'.png')
    image_resized.save(image_resized_dir+img_name+'.jpg')
    gt_resized.save(gt_resized_dir+gt_name+'.png')

    print("done!")





