import torch as T

def accuracy(preds, labels):
    return T.sum(((preds >= 0)*2)-1 == labels.squeeze())/labels.squeeze().shape[0]
