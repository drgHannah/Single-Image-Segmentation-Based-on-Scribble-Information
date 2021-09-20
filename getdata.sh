#!/bin/sh

mkdir ./data/

# download scribble annotations from http://cs.uwaterloo.ca/~m62tang/rloss/
wget http://cs.uwaterloo.ca/~m62tang/rloss/pascal_2012_scribble.zip
unzip pascal_2012_scribble.zip -d ./data/
rm ./pascal_2012_scribble.zip

# download voc 2012 dataset
wget host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar  -C ./data/
rm ./VOCtrainval_11-May-2012.tar

wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
unzip SegmentationClassAug.zip?dl=0 -d ./data/VOCdevkit/VOC2012/
rm ./SegmentationClassAug.zip?dl=0

# download the pretrained network for semantic soft segmentation
wget cvg.ethz.ch/research/semantic-soft-segmentation/SSS_model.zip
unzip ./SSS_model.zip -d ./data/pretrained_model/
rm ./SSS_model.zip