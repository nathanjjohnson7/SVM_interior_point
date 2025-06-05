import torch as T

def accuracy(preds, labels):
    #labels are 1 for positive and -1 for negative
    return T.sum(((preds >= 0)*2)-1 == labels.squeeze())/labels.squeeze().shape[0]
