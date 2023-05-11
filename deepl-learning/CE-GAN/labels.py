import torch
from numpy.random import choice

def noisy_labels(y, p_flip) :
  # determine the number of labels to flip
  n_select = int(p_flip * y.shape[0])
  # choose labels to flip
  flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
  # invert the labels in place
  y[flip_ix] = 1 - y[flip_ix]
  return y

def label_smoothing_valid(label) :
  label = label - 0.1 + (torch.rand(valid.shape,device=dev)*0.1)
  for i in range(batch_size):
    # for flipped 0 values that are negative 
    if label[i]<0.9:
    # multiply -1 to turn positive
        label[i] *=-1 
  return label
def label_smoothing_fake(label) :
  label = label + (torch.rand(valid.shape,device=dev)*0.1)
  for i in range(batch_size):
    # for flipped 1 values that are greater then 1 
    if label[i]>1.0:
        label[i] +=-0.1 
  return label    
