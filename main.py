import os
import sys
sys.path.append('./SIGGRAPH18SSS/')

import torch

import dataset
import models
import regularization
from optimization import Optimizer
import paths


def calculate_in_chn(setting, train_set):
    ''' Returns the number of input channels of the network.'''

    dimsxy = train_set.get_xy_dimension()

    if setting['input'] == 'rgb':
        return 3
    elif setting['input'] == 'rgbxy':
        return 3 + dimsxy
    elif setting['input'] == 'xy':
        return dimsxy
    else:
        print('Error: Wrong Input.')
        exit()


def run(setting, save_name, savedir):

    # Dataset
    voc_dataset = dataset.VOC2012("val", False)
    train_set = dataset.SingleImage(voc_dataset[setting['idx']], setting, 'train')

    # Load Model
    model = models.getmodel(setting, calculate_in_chn(setting, train_set), train_set.get_number_classes())

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=setting['lr'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_set.get_number_classes())

    # Regularization
    regu = regularization.Regularizer(setting['regularization'] ,train_set)

    # Training
    miou, acc, loss = Optimizer.train(model, train_set, criterion, optimizer, 
                                                        regu, setting, os.path.join(savedir, save_name))

    return miou, acc, loss
    

if __name__ == "__main__":

    setting = {
        'dev':"cuda",
        'idx': 43, 
        'xytransform': "xy",
        'name':"CNN_Net",
        'kernel_size': 3,
        'width': 16, 
        'depth': 3,
        'lr':  0.0005,
        'epochs': 3000,
        'bs': None,
        'input': "rgbxy",
        'xytype': "featxy",
        'regularization': "none",
        'tau': 0,
        'xygrad': 0,
        'rgbgrad': 0
    }
    savedir = f"./results/{setting['name']}/image_{setting['idx']}/"
    os.makedirs(savedir, exist_ok=True)
    save_name = "result"
    
    miou, acc, loss = run(setting, save_name, savedir)

    print(f"\nFinished with Mean IoU: {miou}, Accuracy: {acc} and Loss: {loss}.")
