import cv2
import glob


#if __name__ == "__main__"

img_path = '/path/to/datasets/TextSegmentation/total_text/images/Test/'

gt_path = '/path/to/datasets/TextSegmentation/total_text/pixel_GT/Test/'

gt_color_path = '/path/to/datasets/TextSegmentation/total_text/pixel_GT_color/Test/'

#img_list = sorted(glob.glob(img_path+"*.jpg"))
img_list = sorted(glob.glob(img_path+"*.*"))
#gt_list = sorted(glob.glob(gt_path+"*.png"))
#gt_list = sorted(glob.glob(gt_path+"*.jpg"))
gt_list = sorted(glob.glob(gt_path+"*.*"))

#for indx in range(0, len(img_list)):
for indx in range(228, len(img_list)):    
#for indx in range(936,937):
#for indx in range(894,len(img_list)):    
#for indx in range(0, 5):    
    image_filename = img_list[indx]
    gt_filename = gt_list[indx]
    #print image_filename
    #print gt_filename

    name = image_filename.split('/')[-1].split('.')[0]
    print name
    img = cv2.imread(image_filename)
    gt = cv2.imread(gt_filename)
    gt = gt/255

    gt_color = img*gt
    #cv2.imwrite(gt_color_path+name+'.png',gt_color)
    cv2.imwrite(gt_color_path+name+'.jpg',gt_color)

    print "index of query is : {} done!".format(indx)

#gt = cv2.imread('/path/to/datasets/TextSegmentation/GT/img_860.png')