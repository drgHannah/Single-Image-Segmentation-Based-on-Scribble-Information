import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageFilter

import regularization as reg
import utils
from metrics import get_pxacc_miou
from utils import color_map
import paths
import models

class Optimizer():
    ''' Optimization of Scribble based Segmentation.'''


    @staticmethod
    def train(model, data, criterion, optimizer, regularization, setting, save_name):
        ''' Training of scribble based Segmentation.

        Args:
            model:
                Neural Network model used for the training.
            data:
                Single image Dataset of the class "Single_Image".
            criterion:
                Loss function.
            optimizer:
                Optimizer.
            regularization:
                Regularization of the energy function, e.g. total variation.
            setting:
                Dictionary defining the setting for the training procedure.
            save_name:
                Name under which the trainings results are saved.
        Returns:
            The final intersection over union value,
            the final pixel accuracy and the final loss value.
        '''

        model.train()
        loss_plot = []

        # Case1: Training on Pixelwise Networks
        if setting['name'] == "FC_Net":

            # Extract Data
            sample = data['2d']
            img2d = sample['rgb'].to(setting['dev'])
            xy2d = sample['xy'].to(setting['dev'])
            scribble2d = sample['scribble'].squeeze(1).to(setting['dev'])
            gt2d = sample['gt'].flatten()            

            # Train Loop
            for epoch in tqdm(range(setting['epochs'])):
                reg_loss = 0
                model.train()

                # Extract only Scribbled Pixels
                scribbled_pixels = scribble2d!=data.get_number_classes()
                img = img2d[scribbled_pixels]
                scribble = scribble2d[scribbled_pixels]
                gt = gt2d[scribbled_pixels]
                xy = xy2d[scribbled_pixels]

                # Select randomly setting["bs"] many pixels
                if setting["bs"] is not None:
                    no = min(setting["bs"], len(gt)-1)
                    indices = torch.randint(0, len(gt)-1, (no,))
                    img = img[indices]
                    scribble = scribble[indices]
                    gt = gt[indices]
                    xy = xy[indices]
            
                # Zero Grad
                optimizer.zero_grad()

                # Predict
                pred = model(img, xy)
                reg_loss = setting['tau'] * regularization(pred)
                loss = criterion(pred, scribble.long()) + reg_loss

                # Optimization
                loss.backward()
                optimizer.step()

                # Loss and Metrics
                with torch.no_grad():
                    loss_plot.append(loss.item())

                    if epoch == (setting['epochs'])-1:

                        # Get Accuracies
                        model.eval()
                        argmax_pred = torch.argmax(model(img2d, xy2d), dim=1)
                        acc, miou = get_pxacc_miou(gt2d.cpu(), argmax_pred.cpu(), noneclass = data.get_number_classes())

                        # Plot
                        pred = pred.reshape(gt.shape[0],gt.shape[1],pred.shape[1]).permute(2,0,1).unsqueeze(0)
                        Optimizer.plot_state(pred.cpu(), img.cpu(), scribble.cpu(), gt.cpu(), loss_plot, save_name)
                        Optimizer.save_contoured_prediction(torch.argmax(pred[0],dim=0).cpu().numpy(), img[0].detach().cpu(), save_name)

        # Case2: Training on Convolutional Networks
        else:

            # Extract Data
            sample = data['3d']
            img = sample['rgb'].to(setting['dev'])
            xy = sample['xy'].to(setting['dev'])
            scribble = sample['scribble'].to(setting['dev'])
            gt = sample['gt']

            # Train Loop
            for epoch in tqdm(range(setting['epochs'])):

                if setting['input'] != 'rgb':
                    xy.requires_grad = True
                    img.requires_grad = True

                # Zero Grad
                optimizer.zero_grad()

                # Predict
                pred = model(img, xy.float())
                loss = criterion(pred, scribble.long())
                
                # Regularization
                if setting['input'] != 'rgb':
                    
                    input_gradient = torch.autograd.grad(loss, xy, retain_graph=True, create_graph=True)
                    input_gradient_rgb = torch.autograd.grad(loss, img, retain_graph=True, create_graph=True)

                    loss += setting['xygrad'] * torch.mean(torch.abs(input_gradient[0])) * 1e6
                    loss += setting['rgbgrad'] * torch.sum(torch.abs(input_gradient_rgb[0]))

                loss += setting['tau'] * regularization(pred)

                # Optimization
                loss.backward()
                optimizer.step()

                # Metrics
                with torch.no_grad():
                    loss_plot.append(loss.item())

                    if epoch == (setting['epochs'])-1:
                        
                        argmax_pred = torch.argmax(pred.cpu(), dim=1)
                        acc, miou = get_pxacc_miou(gt.cpu(), argmax_pred[0].cpu(), noneclass = data.get_number_classes())

                        Optimizer.save_contoured_prediction(argmax_pred[0].numpy(), img[0].detach().cpu(), save_name)
                        Optimizer.plot_state(pred.cpu(), img.cpu(), scribble.cpu(), gt.cpu(), loss_plot, save_name)
                        
        return miou, acc, loss.item()

    @staticmethod
    def plot_state(prediction, image, scribble , ground_truth, loss_full, savename):
        vmin, vmax = 0, torch.max(scribble)
        blend = utils.blend_image_segmentation(image[0], torch.argmax(prediction, dim=1)[0])

        # Plot Loss
        plt.figure(figsize=(10,4))
        plt.plot(loss_full)
        plt.savefig(savename + '_loss.png')
        plt.close()

        # Plot Original Image
        _, axs = plt.subplots(1,4,figsize=(20,4))
        axs[0].imshow(image[0].permute(1,2,0))
        axs[0].set_title('Original Image')
        axs[1].imshow(blend, vmin=vmin, vmax=vmax)
        axs[1].set_title('Prediction')
        axs[2].imshow(scribble[0], vmin=vmin, vmax=vmax)
        axs[2].set_title('Scribble Data')
        axs[3].imshow(ground_truth, vmin=vmin, vmax=vmax)
        axs[3].set_title('Ground Truth')
        plt.savefig(savename + '_all.png')
        plt.close()

    @staticmethod
    def save_contoured_prediction(prediction, image, save_name):
        image = utils.blend_image_segmentation(image.detach().cpu(), prediction)

        # load segmented image as greyscale
        seg = Image.fromarray(np.uint8((prediction))).convert("L")
        image = np.uint8((np.array(image*255)))

        # draw contours
        colors = [(255,255,0),(0,255,0),(0,0,255),(0,255,255),(255,0,0),(255,0,255)]
        for idx, color in zip(range(len(colors)), colors):
            c_contour = seg.point(lambda p:p==(idx+1) and 255)
            edges = np.array(c_contour.filter(ImageFilter.FIND_EDGES))
            image[np.nonzero(edges)] = color

        # save result
        Image.fromarray(image).save(save_name+'_result.png')
