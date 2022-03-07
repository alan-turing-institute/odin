import pytest
import vne
import numpy as np
import odin

from vne.special import ctf
from vne.dataset import SimulatedDataset
import vne.simulate as simulate
from vne.special.ctf import contrast_transfer_function, convolve_with_ctf

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torch.utils.data import DataLoader

from odin.model import Odin_model

def test_create_model():
	model = Odin_model()
	model.set_optimizer()
	
def test_load_model():

def test_train_model():
	simulate.set_default_font()
