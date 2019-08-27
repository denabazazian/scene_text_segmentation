import os
import glob

#gt_dir = "/path/to/datasets/Challenge2_Task2/Challenge2_Test_Task2_GT/"
res_gt_dir = "/path/to/deepLabV3Plus/deeplabv3plus_pixelWise/RRC_webEvaluation/gt_res/"

gt_files = sorted(glob.glob(res_gt_dir+'*.png'))

for file in gt_files:
    file_name = file.split('/')[-1].split('.')[0]
    img_name = '_'.join(file_name.split("_")[1:])
    res_name = 'res_'+img_name+'.png'
    newName = res_gt_dir+res_name
    os.rename(file, newName)