import argparse
import os
import sys
import numpy as np
import pdb
from tqdm import tqdm
import cv2
import glob

import numpy as np
from numpy import *
import matplotlib
#matplotlib.use("Agg")
#matplotlib.use("wx")
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *   
from PIL import Image

# class load_data(Dataset):
#     def __init__(self,args,img_path):
#         super().__init__()
#         self.args = args
#         self.img_path = img_path
                   
#     def __getitem__(self,img_path):
#         image = Image.open(self.img_path).convert('RGB')
#         image = np.array(image).astype(np.float32).transpose((2, 0, 1))
#         image = torch.from_numpy(image).float()

#         return image


def get_model(nclass,args):
    
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    # Using cuda
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()

    checkpoint = torch.load(args.resume)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    return model


def get_pred(img_path,model,args):    
    model.eval() 
    image = Image.open(img_path).convert('RGB')
    #image = image.resize((512,512), Image.ANTIALIAS)
    image = np.array(image).astype(np.float32).transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()

    if args.cuda:
        image = image.cuda()
    with torch.no_grad():
        output = model(image)
    
    #pdb.set_trace()
    # normalize = nn.Softmax(dim=1)
    # output = normalize(output)

    pred = output.data.cpu().numpy()    
    return pred     

def F1_loss(pred,target):

    N = np.logical_or(pred,target)  # logical
    Tp = np.logical_and(pred,target)
    Fn = np.subtract(target,Tp) # element-wise subtraction in pytorch 
    #Fn = np.bitwise_xor(target,Tp)
    Fp = np.subtract(pred,Tp)     
    Tn = np.subtract(N,np.logical_or(Tp,Fp,Fn))

    #pdb.set_trace()
    precision = np.sum(Tp)/(np.sum(Tp)+np.sum(Fp))
    recall = np.sum(Tp)/(np.sum(Tp)+np.sum(Fn)) 
    F1 = (2*np.sum(Tp))/(2*np.sum(Tp)+np.sum(Fn)+np.sum(Fp))
    #F1 = np.true_divide(np.add(2*Tp,Fn,Fp),2*Tp)
    #F1 = np.true_divide(np.sum(np.multiply(2,Tp),Fn,Fp),np.multiply(2,Tp))
    #F1 = np.true_divide(np.multiply(2,Tp),np.multiply(np.sum(Tp,Fn),np.sum(Tp,Fn)))
    #accuracy = np.true_divide(np.add(Tp,Tn),np.add(Tp,Tn,Fp,Fn))
    accuracy = np.sum(Tp+Tn)/np.sum(N)

    return F1 , accuracy, precision, recall

def F1_rwi(pred,target):
    #pred = pred[:,:,0] # using only the red channel
    #target = target[:,:,0]

    N = np.logical_or(pred, target) # logical
    Tp = np.logical_and(pred, target)
    Fn = np.bitwise_xor(target, Tp) # element-wise subtraction in pytorch
    Fp = np.bitwise_xor(pred, Tp)
    xx= np.logical_or(np.logical_or(Tp,Fp), Fn)
    Tn = np.bitwise_xor(N, xx)

    precision = Tp.sum()/(Tp.sum()+ Fp.sum() )
    recall = Tp.sum()/(Tp.sum()+ Fn.sum())
    F1 = 2*Tp.sum() /(2*Tp.sum()+ Fn.sum()+ Fp.sum())
    accuracy = (Tp.sum()+Tn.sum())/N.sum()

    return F1, accuracy, precision, recall


if __name__=='__main__':

    #### Parameters and paths:
    nclass = 2
    save_rrc_res_path = "/path/to/deepLabV3Plus/deeplabv3plus_pixelWise/results/validation_images/B_260/"
    model_path = "/path/to/deepLabV3Plus/deeplabv3plus_pixelWise/results/icdar_models/run/icdar/deeplab-resnet/model_best.pth.tar"
    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@" 
    img_path = "/path/to/GAN_text/data/text_segmentation/test/A/"
    gt_path = "/path/to/GAN_text/data/text_segmentation/test/B_gt_1chanel/"

    ### args
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Heatmap Prediction")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')

    ##checking point
    parser.add_argument('--resume', type=str, default= model_path,
                        help='put the path to resuming file if needed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
                        
    image_files = sorted(glob.glob(img_path+'*.png')) #'*.jpg'))

    trained_model = get_model(nclass,args)

    f1_all = []
    accuracy_all = []

    f1_all_rwi = []
    accuracy_all_rwi = []

    #for img_path in sys.argv[1:]:
    #for i in range(0,10):
    for i in range(0,len(image_files)):

        img_path = image_files[i]
        print("image path is: {}".format(img_path))

        img_name = img_path.split('/')[-1].split('.')[0]

        gt = asarray(Image.open(gt_path+img_name+'.png'))
        #trained_model = get_model(nclass,args)
        #pdb.set_trace()
        
        # load_test_data = load_data(args,img_path)
        # dataloader = DataLoader(load_test_data)
        
        # for ii, img_test in enumerate(dataloader):
        pred = get_pred(img_path,trained_model,args)

        pred = softmax(pred, axis=1)
        #image_source = cv2.imread(img_path)
        #image_source = cv2.resize(image_source, (512, 512))
        #pdb.set_trace()

        #fig = plt.figure()
        # plt.imshow(pred.squeeze()[1,:,:])
        # plt.show()
        # res = pred.squeeze()[1,:,:]>0.3
        #res = np.argmax(pred.squeeze(), axis=0)
        #pdb.set_trace()
        # plt.imshow(res)
        # plt.show()
        ret,pred_bin = cv2.threshold(pred.squeeze()[1,:,:],0.2,255,cv2.THRESH_BINARY)

        #pdb.set_trace()
        f1, acc, prc, rcl = F1_loss(pred_bin>5,gt>5)
        print("F1 is {}, accuracy is {}, precision is {}, recall is {}".format(f1,acc,prc,rcl))

        #pdb.set_trace()
        pred_bin_8 = pred_bin.astype(np.uint8)
        f1_rwi, acc_rwi, prc_rwi, rcl_rwi = F1_rwi(pred_bin_8>5,gt>5)
        print("F1_rwi is {}, accuracy_rwi is {}, precision_rwi is {}, recall_rwi is {}".format(f1_rwi,acc_rwi,prc_rwi,rcl_rwi))

        f1_all.append(f1)
        accuracy_all.append(acc)

        f1_all_rwi.append(f1_rwi)
        accuracy_all_rwi.append(acc_rwi)

    print("the average of F1 is {}".format(np.mean(f1_all)))    
    print("the average accuracy is {}".format(np.mean(accuracy_all)))    

    print("the average of F1_rwi is {}".format(np.mean(f1_all_rwi)))    
    print("the average accuracy_rwi is {}".format(np.mean(accuracy_all_rwi)))