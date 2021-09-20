import numpy as np
import os
import torch
import torch.nn as nn


class Regularizer():
    ''' Calls desired regularization class.

    Calls a the desired regularization. This class contains 
    so far only Total Variation and can be extended with 
    further regularizations.
    '''


    def __init__(self, rtype, train_set):
        ''' Initializes the Regularizer Class.

        Args:
            rtype: 
                The regularization name. E.g. "tv" for calling "Total Variation".
            train_set:
                "Single_Image" dataset object.
        '''

        self.rtype = rtype
        self.reg = {
            'tv':0
        }

        if "tv" in rtype:
            self.reg['tv'] = TV(orig_image=train_set['3d']['rgb'][0])

    def __call__(self, pred):
        ret = 0
        div = 0
        if "tv" in self.rtype:
            ret += self.reg['tv'](pred)
            div+=1
        
        if div>0:
            return ret / div
        else:
            return 0


# TV
class TV(nn.Module):
    ''' Total Variation.'''


    def __init__(self, orig_image=None):
        ''' Initialization.
    
        args:
            orig_image: Used for calculating the weighting of tv based on image values. 
            If orig_image=None, no weighting is applied.
        '''

        super(TV,self).__init__()
        self.img = orig_image.unsqueeze(0)

    def forward(self,x):
        ''' Returns TV of the 4D (batch size, channels, height, width) input image x.'''

        # TV
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        # Weight
        loss_weight = 1
        if self.img is not None:
            gamma = 5
            g_img = torch.mean(self.img, dim=1)
            h_tv_img = torch.pow((g_img[:,1:,:]-g_img[:,:-1,:]),2).sum()
            w_tv_img = torch.pow((g_img[:,:,1:]-g_img[:,:,:-1]),2).sum()
            deriv_img = (torch.abs(h_tv_img/count_h) + torch.abs(w_tv_img/count_w))/batch_size
            loss_weight = torch.exp(-gamma * deriv_img)/2

        return loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
