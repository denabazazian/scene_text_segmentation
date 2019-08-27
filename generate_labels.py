# python generate_labels.py /path/to/dataset/ICDAR2015/ch4_train_img/img_49.jpg

#import matplotlib.pyplot as plt
#from skimage.draw import line, polygon, circle, ellipse
import numpy as np
import sys
import re
import cv2

gt_png_path = '/path/to/dataset/ICDAR2015/ch4_gt_img_train/'
gt_txt_path = '/path/to/dataset/ICDAR2015/ch4_train_gt/'

img_id = 0 
for img_file in sys.argv[1:]:
    print(img_file)
    img_id +=1

    img_name = img_file.split('.')[0].split('/')[-1]

    img = cv2.imread(img_file)
    cv_size = lambda img: tuple(img.shape[1::-1])
    width_main, height_main = cv_size(img)

    #gt_png = img = np.zeros(width_main, height_main)
    gt_png = np.zeros([height_main,width_main],dtype=np.uint8)

    #read ground truth 
    #gt = open(img_name.split('img')[0]+'voc_'+(img_name.split('/')[-1].split('.')[0])+'.txt').read()   # for the dictionary and distractor
    gt = open(gt_txt_path+img_name+'.txt').read()
    
    numGTs = gt.count('\n')
    #print('numGT is {}'.format(numGTs))
    gt = re.sub(r'[^\x00-\x7f]', r'', gt)
    #print("gt is befor split :{}".format(gt))
    #gt = gt.split('\r\n')
    gt = gt.split('\n')
    #print("gt is after split :{}".format(gt))
    idgt = 0

    # Read the text proposals of the correspond image
    

    for words in range(0, numGTs):
        #if (len(gt[words]) > 0 and gt[words].split(',')[-1].strip() != '###'):
        #print("len GT[word] is: {}".format(len(gt[words])))
        if (len(gt[words]) > 0):
            idgt += 1
            gt_points = np.array( [[[gt[words].split(',')[0],gt[words].split(',')[1]],
                                  [gt[words].split(',')[2],gt[words].split(',')[3]],
                                  [gt[words].split(',')[4],gt[words].split(',')[5]],
                                  [gt[words].split(',')[6],gt[words].split(',')[7]]]],
                                  dtype=np.int32 )
            
            cv2.fillPoly(gt_png, gt_points, 255 )
            #print(idgt)
    # write image
    cv2.imwrite(gt_png_path + img_name + '.png', gt_png)
    print("img_id {} done!".format(img_id))
