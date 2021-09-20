import os
import warnings
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from PIL import Image

import paths
from transformator import Transformator
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from main_hyper_copy import main as hypermain


class VOC2012(Dataset):
    ''' Visual Object Classes 2012 (VOC2012) Dataset

    This class reads the VOC2012 images and their ground truth,
    as well as the scribble masks  from the given paths. The pathes
    should be set in the file path.py. 
    '''


    def __init__(self, mode = "train", transform = True, semantic = False):
        ''' Initializes the class VOC2012.

        Args:
            mode:
                The mode can take the values: "train", "val" and "trainval", that 
                specify whether the training data, test data or validation data will be loaded.
            transform:
                A boolean, indicating if applying data augmentation to the data.
            semantic:
                A boolean indicating if the ground truth is dependend on the semantic meaning of the object.
        '''

        self.semantic = semantic

        # Directories
        root_dir = paths.get_path()["root_voc_dir"]
        name_dir  = os.path.join(root_dir, 'ImageSets', 'Segmentation')
        self.mask_dir = paths.get_path()["scribble_dir"]
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.gt_dir = os.path.join(root_dir, 'SegmentationClassAug')

        # Load Names
        file = open(os.path.join(name_dir,"{}.txt".format(mode)), 'r')
        self.img_names = file.read().splitlines()
        file.close()

        self.patch_size = 300
        self.transform = transform
        self.dataset_len = len(self.img_names)
        
    def __len__(self):
        '''
        Returns:
            The number of images in the dataset.
        '''

        return self.dataset_len

    def __getitem__(self, idx):
        ''' Loads an image of VOC2012.

        Args:
            idx: 
                Index of the image that should be loaded.

        Returns: 
            A dict including image data of one image and its ground truth, mask and name.
        '''

        # Load Image and Ground Truth
        image = Image.open(self.img_dir + '/' + self.img_names[idx] + '.jpg')
        ground_truth = Image.open(self.gt_dir + '/' + self.img_names[idx] + '.png')
        mask = np.array(Image.open(self.mask_dir + '/' + self.img_names[idx] + '.png'))
        mask[mask<21] = 0
        mask[mask>=21] = 1
        mask = (1-mask)

        # Set Augmentation
        if self.transform:
            image, ground_truth, mask = self.data_augmentation(image, ground_truth.convert('P'), mask)
            ground_truth = torch.tensor(np.array(ground_truth)).int()
            mask = torch.tensor(np.array(mask)).int()
        else:
            trans = torchvision.transforms.ToTensor()
            image = trans(image)
            ground_truth = torch.tensor(np.array(ground_truth)).int()
            mask = torch.tensor(mask).int()

        ground_truth[ground_truth == 255] = 21

        if self.semantic == False:
            ground_truth = self.remove_semantic_information(ground_truth)
        
        sample = {'image':image, 'label':ground_truth, 'mask':mask, 'name':self.img_names[idx]}
        return sample

    def data_augmentation(self, img, gt, mask_b):
        """ Applies augmentation to the input data.

        This function applies random horizontal flipping, random rotation, 
        color jitter and noise to the data.
        
        Args:
            img: 
                VOC2012 data image.
            gt:
                Ground truth image.
            mask_b:
                Binary scribble mask.

        Returns:
            Augmented image, ground truth and scribble mask.
        """

        mask = Image.fromarray(mask_b,'P')
        
        # Random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            gt = TF.hflip(gt)

        # Random Rotation
        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
            gt = TF.rotate(gt, angle)

        trans = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                torchvision.transforms.ToTensor(),
                ])
        
        img = trans(img)

        # Add Noise
        img = img + torch.randn_like(img) * 0.05
        img = torch.clamp(img, min=0, max=1)

        return img, gt, mask

    def remove_semantic_information(self, image):
        """ Removes the semantic information of the input
        (the input should be thr ground truth data).
        """
        vals = torch.unique(image)
        for i in range(len(vals)):
            image[image==vals[i]]=i
            print("changed class %.3d --> %.2d"%(vals[i].item(),i))
        return image


class SingleImage():
    ''' Dataset consisting of a single image.

    This class prepares an input image, ground truth and scribble mask
    for training.
    '''


    def __init__(self, sample, settings, mode):
        ''' Initializes the class Single_Image.

        Args:
            mode:
                The mode can take the values: "train", or something different, that 
                specify whether the training data, test data or validation data will be loaded.
                This data is prepared differently.
            sample:
                A dict including at least image data of one image ("image") and its ground truth ("label")
                and a scribble mask ("mask").
            settings:
                A dict including the feature that will be concatenated to the image for the training.
                settings["xytype"] can have the values: "feat", "featxy" or "xy".
        '''

        # Load Data
        self.gt = sample["label"]
        self.mask = sample["mask"]
        self.image = sample["image"]
        noneclass = torch.max(self.gt)
        self.scribble = (self.mask * self.gt + (1-self.mask) * noneclass)

        feat_path = os.path.join(paths.get_path()["feature_dir"], sample['name']+ '.pt')
        try:
            self.feat = torch.load(feat_path)
        except:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hypermain(sample['name'])
            self.feat = torch.load(feat_path)

        # Image Info
        c,h,w = self.image.shape
        self.nr_pixel = w*h

        # Additional Settings
        self.settings = settings

        # Create Grid
        self.xy = Transformator.get_positional_matrices(w, h)
        self.xy = Transformator.get_transformation_by_name(settings['xytransform'],self.scribble,self.xy)

        # Get Positions - Training: scribble positions - Validation: all coordinates
        if mode=='train':
            self.idx_h, self.idx_w = torch.where(self.scribble != noneclass)
        else:
            xxm, yym = torch.meshgrid(torch.arange(0, self.height, 1), torch.arange(0, self.width, 1))
            self.idx_h, self.idx_w = xxm.reshape(-1),yym.reshape(-1)

        # Apply PCA to the semantic features
        n_components = 2
        inter_rep = torch.tensor(self.feat['embedmap'])
        w,h,c = inter_rep.shape
        inter_rep = inter_rep.permute(2,0,1)
        X = inter_rep.reshape([c,-1])
        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Reshape Data
        feat = torch.tensor(pca.components_).reshape([n_components, w, h]).unsqueeze(0).float()
        self.feat = (feat - torch.min(feat)) / (torch.max(feat) - torch.min(feat))

        if settings['xytype'] == "feat":
            self.xy = feat.squeeze(0)
        elif settings['xytype'] == "featxy":
            self.xy = torch.cat((self.xy,feat.squeeze(0)),dim=0)

    def get_number_classes(self):
        '''Returns the number of classes in the current image.'''

        return torch.max(self.gt).item()

    def get_xy_dimension(self):
        ''' Returns the feature dimension.'''

        return self.xy.shape[0]

    def __getitem__(self, dimensional="3d"):
        ''' Loads the image and its features.

        Args:
            dimensional: 
                Can be "2d" or "3d" and decides if the output image has the shape (pixels x 3) for "2d"
                or (1 x 3 x width x height) for "3d".

        Returns: 
            A dict including image data of one image and its ground truth and its mask, the features.
        '''
        
        xy_chn = self.xy.shape[0]
        if dimensional == "2d":
            rgb = self.image.reshape([3,-1]).permute(1,0) # pixel(batchsize) x rgb(3)
            xy = self.xy.reshape([xy_chn,-1]).permute(1,0) # pixel(batchsize) x xy_channels(standard: 2)
            scribble = self.scribble.flatten().unsqueeze(1) # 1 x scribble-label
            gt = self.gt.flatten().unsqueeze(0) # 1 x scribble-label
        if dimensional == "3d":
            rgb = self.image.unsqueeze(0) # 1 x 3 x w x h
            xy = self.xy.unsqueeze(0) # 1 x xy_chn x w x h
            scribble = self.scribble.unsqueeze(0) # 1 x w x h
            gt = self.gt
        return {"rgb": rgb.to(self.settings['dev']),
                "xy":xy.to(self.settings['dev']),
                "scribble":scribble.to(self.settings['dev']),
                "gt": gt.to(self.settings['dev']),
                "mask":self.mask.to(self.settings['dev']),
                "feat":self.feat.to(self.settings['dev']),
                }
