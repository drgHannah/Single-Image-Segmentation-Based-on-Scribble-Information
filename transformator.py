import numpy as np
import torch
import scipy
import scipy.stats as st


class Transformator():
    ''' A collection of operations on the positional image encoding.'''


    @staticmethod
    def get_transformation_by_name(names, scribble, xy, filename = ""):
        h=[]
        if "distance_scribble" in names:
            h.append( Transformator.distance_scribble(scribble))
        if "gauss_bubbles" in names:
            h.append(Transformator.gauss_bubbles(xy))
        if "xy" in names:
            h.append(xy)

        return torch.cat(h, dim=0)

    @staticmethod
    def get_positional_matrices(w,h):
        ''' Calculates meshgrid of coordinates with values in [0,1].

            Args:
                w: width
                h: height

            Returns: grid(tensor: 2, w, h) 
        '''

        x = torch.linspace(0,1, h)
        y = torch.linspace(0,1, w)
        xx, yy = torch.meshgrid(x, y)
        grid = torch.stack((yy, xx), axis=0).float()
        return grid

    @staticmethod
    def distance_scribble(scribble):
        ''' Calculates distance of each pixel to each scribble.

            Args:
                scribble (tensor: w, h): 2 dimensional scribble, with maximal class being none-class

            Returns: grid(tensor: number of scribbles, w, h)
        '''

        w,h=scribble.shape
        c=torch.max(scribble)

        scribble_oh = torch.nn.functional.one_hot(scribble.long())
        grid = torch.zeros(c,w,h)

        for i in range(c):
            dist = scipy.ndimage.morphology.distance_transform_edt(1-scribble_oh[:,:,i].cpu(), sampling=[1/w, 1/h])
            grid[i] = torch.tensor(dist)

        return grid

    @staticmethod
    def gauss_bubbles(xy):
        ''' Multiplies xy with gauss bubbles.

            Args:
                xy (tensor: 2, w, h): positional coordinates

            Returns: grid(tensor: number of scribbles, w, h)
        '''

        no=20
        _, w, h=xy.shape
        result = torch.zeros(no, w, h) * 1.0

        kernel_size = 101
        std = 3

        for i in range(no):
            x = torch.randint(kernel_size//2,w-kernel_size//2-1,(1,1))[0][0]
            y = torch.randint(kernel_size//2,h-kernel_size//2-1,(1,1))[0][0]
            gauss_kernel = torch.tensor(Transformator.gkern(kernel_size,std)) * 500
            result[i, (x-kernel_size//2):(x+kernel_size//2+1), (y-kernel_size//2):(y+kernel_size//2+1)] = gauss_kernel



        return result

    @staticmethod
    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel."""

        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()
