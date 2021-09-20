import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import jaccard_score


def calculate_confusion_matrix(true, pred):
    '''Calculates confusion matrix. Shape of input can be height x width or height * width.'''

    #Input Shape: H x W, or H*W
    assert true.shape == pred.shape
    num_classes = torch.max(true)+1
    true_oh = (torch.nn.functional.one_hot(true.to(torch.int64),num_classes=num_classes)) # w x h x cl
    pred_oh = (torch.nn.functional.one_hot(pred.to(torch.int64),num_classes=num_classes)) # w x h x cl
    classes = true_oh.shape[-1]
    true_oh = true_oh.view(-1, classes)
    pred_oh = pred_oh.view(-1, classes)
    true_oh = true_oh[:,:-1]
    pred_oh = pred_oh[:,:-1]
    mcm = multilabel_confusion_matrix(np.array(true_oh), np.array(pred_oh))
    return mcm


def mcm_to_val(mcm):
    tp = mcm[:,1,1]
    fp = mcm[:,1,0]
    fn = mcm[:,0,1]
    tn = mcm[:,0,0]
    return tp,fp,fn,tn   


def get_accuracy(mcm):
    tp, fp, fn, tn = mcm_to_val(mcm)
    return (tp+tn)/(tp+fp+tn+fn)

def get_iou(mcm):
    tp, fp, fn, tn = mcm_to_val(mcm)
    return tp / (tp + fn + fp)


def get_precision(mcm):
    tp, fp, fn, tn = mcm_to_val(mcm)
    return tp / (tp + fp)


def get_recall(mcm):
    tp, fp, fn, tn = mcm_to_val(mcm)
    return tp / (tp + fn)


def get_pixel_accuracy(true, pred, noneclass = None):
    ''' Returns accuracy. Noneclass isn't taken into account.'''

    if noneclass is not None:
        mask = true!=noneclass
        true = true[mask]
        pred = pred[mask]
    bin = (true == pred)
    return sum(bin * 1.0) / len(bin * 1.0)


def get_pxacc_miou(true, pred, noneclass):
    ''' Returns accuracy and mean intersection over union. Noneclass isn't taken into account.'''

    pxacc = np.array(get_pixel_accuracy(true.flatten(), pred.flatten(), noneclass))
    iou = jaccard_score(true[true!=noneclass],pred[true!=noneclass], average=None)
    return (pxacc), (np.mean(iou))
