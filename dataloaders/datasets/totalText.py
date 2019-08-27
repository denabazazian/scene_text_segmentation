from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from numpy import *

class TotalTextSegmentation(Dataset):
    """
    totalText dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('totalText'),
                 split='train',
                 ):
        """
        :param base_dir: path to totalText dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir,'images/Train')         
        self._cat_dir = os.path.join(self._base_dir, 'pixel_GT/Train') 
        # self._image_dir = os.path.join(self._base_dir, 'ch4_train_img_jpg')
        # self._cat_dir = os.path.join(self._base_dir, 'ch4_train_gt_png')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        #_splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg") #".png")
                print("img is: {}".format(_image))
                _cat = os.path.join(self._cat_dir, line + ".jpg") #".png")
                print("gt is: {}".format(_cat))
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)                
            elif split == 'val':
                return self.transform_val(sample)
        #return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((640,480), Image.ANTIALIAS)  #*** NOT already resized images *****
        #_img = _img.resize((512,512), Image.ANTIALIAS)
        #width_img, height_img = _img.size
        #print("img width is: {}, img height is: {}".format(height_img,width_img))
        #_target = Image.open(self.categories[index])
        _target= asarray(Image.open(self.categories[index]))
        #print("category index is: {}".format(self.categories[index]))
        _target_255 = _target/255
        _target_img = Image.fromarray(_target_255)
        _target_resized = _target_img.resize((640,480), Image.ANTIALIAS) #*** NOT already resized images *****
        #_target = _target.resize((512,512), Image.ANTIALIAS)
        #_target = _target/255
        #width_gt, height_gt = _target.size
        #print("GT width is: {}, GT height is: {}".format(height_gt,width_gt))

        return _img, _target_resized

    def transform_tr(self, sample):
        # composed_transforms = transforms.Compose([
        #     tr.RandomHorizontalFlip(),
        #     tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
        #     tr.RandomGaussianBlur(),
        #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     tr.ToTensor()])
        composed_transforms = transforms.Compose([tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        # composed_transforms = transforms.Compose([
        #     tr.FixScaleCrop(crop_size=self.args.crop_size),
        #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     tr.ToTensor()])
        composed_transforms = transforms.Compose([tr.ToTensor()])

        return composed_transforms(sample)

    # not sure is I can comment it?!?!
    # def __str__(self):
    #     return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    icdar_train = ICDARSegmentation(args, split='train')

    dataloader = DataLoader(icdar_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='icdar')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


