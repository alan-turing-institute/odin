import vne
import numpy as np
import shutil

from vne.special import ctf
from vne.dataset import SimulatedDataset
import vne.simulate as simulate

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torch.utils.data import DataLoader

def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)

    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)
        
def load_checkpoint(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch']#, valid_loss_min.item()


# Add this to VNE?
def set_font(path:"./fonts/OpenSans-Regular.ttf"):
	"""
	path: path to the font
	"""
	simulate.set_default_font(path, 20)

def show_image():
	"""
	plot the images with or without the bounding boxes
	"""