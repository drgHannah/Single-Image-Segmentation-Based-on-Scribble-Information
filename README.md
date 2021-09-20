## Learning or Modelling? An Analysis of Single Image Segmentation Based on Scribble Information

This repository is the official implementation of [Learning or Modelling? An Analysis of Single Image Segmentation Based on Scribble Information](https://ieeexplore.ieee.org/abstract/document/9506185/).


**Single image segmentation based on scribbles that rely
on ...**
<table>
   <tr>
      <td><img src="https://github.com/drgHannah/Scribblebased-Image-Segmentation/blob/main/43-gt.png "width = 360px></td>
      <td><img src="https://github.com/drgHannah/Scribblebased-Image-Segmentation/blob/main/43-rgb.png "  width = 360px></td>
      <td><img src="https://github.com/drgHannah/Scribblebased-Image-Segmentation/blob/main/43-grad-xy-reg.png " width = 360px></td>
      <td><img src="https://github.com/drgHannah/Scribblebased-Image-Segmentation/blob/main/43-grad-feat-reg.png "  width = 360px></td>
  </tr>
   <tr>
      <td>  </td>
     <td>... image color, using a simple CNN</td>
      <td> ... image color and spacial information </td>
     <td>... image color and semantic information</td>
  </tr>
</table>

### Abstract
Single image segmentation based on scribbles is an important technique in several applications, e.g. for image editing software. In this paper, we investigate the scope of single image segmentation solely given the image and scribble information using both convolutional neural networks as well as classical model-based methods, and present three main findings: 1) Despite the success of deep learning in the semantic analysis of images, networks fail to outperform model-based approaches in the case of learning on a single image only. Even using a pretrained network for transfer learning does not yield faithful segmentations. 2) The best way to utilize an annotated data set is by exploiting a model-based approach that combines semantic features of a pretrained network with the RGB information, and 3) allowing the networks prediction to change spatially and additionally enforce this variation to be smooth via a gradient-based regularization term on the loss (double backpropagation) is the most successful strategy for pure single image learning-based segmentation.



## Get Started
- **Data** \
To download the VOC2012 dataset, associated scribbles and the pretrained neural network used for the  semantic features, please run the shell script *getdata.sh*.
- **Semantic Features** \
To use the semantic features from *Semantic Soft Segmentation* of Aksoy et. al\*, clone the following repository in the project folder:
[https://github.com/iyah4888/SIGGRAPH18SSS](https://github.com/iyah4888/SIGGRAPH18SSS)
- **Dependencies** \
Please create an Anaconda environment and install the requirements from the . yml file. Note that you need at least Python 3.X.

\*Yaugiz Aksoy, Tae-Hyun Oh, Sylvain Paris, Marc Pollefeys, and Wojciech Matusik 2018. Semantic Soft Segmentation. _ACM Transactions on Graphics (Proc. SIGGRAPH), 37_(4), p.72:1-72:13.

## Information on the setting:
The parameters of the experiments can be set in main.py:
- idx: Index of the image to be segmented. 
-  Network settings:
	- name: The network used for segmentation can take "CNN_Net ",  "FC_Net ",  "DenseNet" and "UNet ".
	- kernel_size: Kernel size of the CNN.
	- width: Network width of the CNN, the Fully Connected Network or the U-Net.
	- depth: Network width of the CNN or the Fully Connected Network.
- Training parameters:
	- lr: Learning rate.
	- epochs: Minimization steps in the optimization.
	- bs: Batchsize, only used for the Fully Connected Network. Can be set to "None".
- Regularization and input:
	- input: Specifies whether the network input contains only the color image ("rgb") or the color image and additional information ("rgbxy").
	- xytype: Specifies what kind of additional information is involved: spatial information ("xy"), semantic features ("feat"), or both ("featxy").
	- regularization: Additional regularization, e.g. "tv" for total variation.
	- xygrad: Regularization of the influence of the additional network input.
	- rgbgrad: Regularization of the influence of the input color image.

