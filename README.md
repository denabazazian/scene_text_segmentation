# scene_text_segmentation
Pixel-wise scene text segmentation based on DeepLabV3+ [paper](https://arxiv.org/pdf/1802.02611.pdf) and its Pytorch [implementation](https://github.com/jfzhang95/pytorch-deeplab-xception). 

## Results ##
Qualitative results of English (first four columns) from ICDAR2013 dataset and Korean (fifth trough eighth columns) from KAIST
dataset. <br/>
Korean text has been segmented in zero-shot learning, the trained models have never seen the Korean text images.

![](fig/textSeg.png)

### Installation ###

Create a conda environmet by installing following packages:

```
conda install python=3.6 ipython pytorch=0.4 torchvision opencv=3.4.4 tensorboardx mkl=2019 tensorboard tensorflow tqdm scikit-image
```
* Required packages:
    * Pytorch 0.4
    * OpenCV 3.4.4
    * mkl 2019
    * tqm
    * scikit-image
    * tensorboardX

### Train ###
The path for training dataset should be defined in ``` mypath.py ```. Then, for instance for ICDAR dataset in ```dataloaders/datasets``` the ```icdar.py``` refers to that. <br/>
* For training ICDAR: <br/>
```
bash train_icdar.sh
```

### Test ###
* For visualizing the heatmaps: <br/>
```
visual_hm.py
```
* For saving the binary text segmentations: <br/>
```
test_save_binary.py
```
* For computing the F1 accuracy: <br/>
```
F1_accuaracy_rwi.py
```


## Citation ##

Please cite [this work](http:) in your publications if it helps your research: <br />

@article{Rawi19,<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	author = {Mohammed Al-Rawi and Dena Bazazian and Ernest Valveny},<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	title = {Can Generative Adversarial Networks Teach Themselves Text Segmentation?},<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	journal = {IEEE Proceedings of International Conference on Computer Vision Workshops},<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	year = {2019}<br />
}<br />
