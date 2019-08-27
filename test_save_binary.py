# python test_save_binary.py /path/to/img_*.jpg

import argparse
import os
import sys
import numpy as np
import pdb
from tqdm import tqdm
import cv2
import glob

import numpy as np
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

if __name__=='__main__':

    #### Parameters and paths:
    nclass = 2
    save_res = "/path/to/saveRes"
    model_path = "/path/to//deeplab-resnet/model_best.pth.tar"
    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@" 
    img_path = "/path/to/input_img/"

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
                        
    image_files = sorted(glob.glob(img_path+'*.png')) #'*.png')) #'*.jpg'))

    trained_model = get_model(nclass,args)

    #for img_path in sys.argv[1:]:
    #for i in range(0,11):
    for i in range(0,len(image_files)):

        img_path = image_files[i]
        print("image path is: {}".format(img_path))

        img_name = img_path.split('/')[-1].split('.')[0]
        #trained_model = get_model(nclass,args)
        #pdb.set_trace()
        
        # load_test_data = load_data(args,img_path)
        # dataloader = DataLoader(load_test_data)
        
        # for ii, img_test in enumerate(dataloader):
        pred = get_pred(img_path,trained_model,args)

        pred = softmax(pred, axis=1)
        #image_source = cv2.imread(img_path)
        #image_source = cv2.resize(image_source, (512, 512))
        # pdb.set_trace()

        #fig = plt.figure()
        # plt.imshow(pred.squeeze()[1,:,:])
        # plt.show()
        # res = pred.squeeze()[1,:,:]>0.3
        #res = np.argmax(pred.squeeze(), axis=0)
        #pdb.set_trace()
        # plt.imshow(res)
        # plt.show()
        
        ret,img_bin = cv2.threshold(pred.squeeze()[1,:,:],0.2,255,cv2.THRESH_BINARY)
        cv2.imwrite(save_res+img_name+".png",img_bin)

        #plt.imshow(pred.squeeze()[i,:,:], cmap='seismic')
        #plt.imshow(pred.squeeze()[i,:,:], vmin=0, vmax=1)
        #plt.imshow(image_source,alpha=.5)
        #plt.imshow(image_source/255,alpha=.5)
        #plt.colorbar()
        #plt.title(('%d,%s')%(i,alphabet[i]))
        #plt.axis('off')
        # fig.savefig(save_rrc_res_path + "res_"+img_name+".png")
        # plt.close(fig)
        
        #for i in range(0,38):
        # for i in range(0,2):

        #     fig = plt.figure()
        #     plt.imshow(pred.squeeze()[i,:,:], cmap='seismic')
        #     #plt.imshow(pred.squeeze()[i,:,:], vmin=0, vmax=1)
        #     #plt.imshow(image_source,alpha=.5)
        #     plt.imshow(image_source/255,alpha=.5)
        #     #plt.colorbar()
        #     plt.title(('%d,%s')%(i,alphabet[i]))
        #     plt.axis('off')
        #     fig.savefig(save_hm_path + "hm_{}_{}.png".format(i, alphabet[i]), dpi=400, bbox_inches='tight')
        #     plt.close(fig)