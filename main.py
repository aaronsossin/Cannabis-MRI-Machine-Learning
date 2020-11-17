import numpy as np
import monai
import glob
import shutil
import tempfile
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadNiftid, RandRotate90d, Resized, ScaleIntensityd, ToTensord
from monai.config import print_config
from monai.data import ArrayDataset, GridPatchDataset, create_test_image_3d
from monai.utils import first
from monai.data import CSVSaver
import os
from glob import glob
import pandas as pd
from run_tests import run_tests
from plot_participant_data import plot_participant_data
from nilearn import datasets, image

"""
Below are a host of Parameters which should be set such as what type of experiment is to be run,
with what model, and with what hyper-parameters.

When ready, run code with "python main.py" in terminal
"""

#To Run Model set to True
run_experiment = True

# Exclamation
print("~...Beep Boop Beep...~")

# TASKS:
## 1. 'classification'
## 2. 'regression'
TASK = "regression"
assert TASK in ["regression","classification"]

# MODELS:
## 1. 'PyTorch'
## 2. 'SpaceNet'
## 3. 'SVM'
## MORE TO COME
MODEL_TYPE = "PyTorch"
assert MODEL_TYPE in ["PyTorch", "SpaceNet", "SVM"]

SPACENET_PENALTY = "tv-l1" #Only  for ni-learn, either "graph-net" or "tv-l1"
assert SPACENET_PENALTY in ["tv-l1", "graph-net"]

# PYTORCH MODELS
## 1. 121
## 2. 201
## 3. 169
## 4. 264
## 5. 1 ~(AlexNet)
## 6. 2 ~(ResNet)
PYTORCH_VERSION = 264
assert PYTORCH_VERSION in [121, 201, 169, 264, 1, 2]

#Whether to use Pre-trained Resnet or not
PRETRAINED_RESNET = False

"""
Note: The 'nilearn' regression also performs segmentation, but different process than the explicit segmenation
"""

# SUBSET:
## 1. FU (only follow up scores (after the 3 years)
## 2. BL (baseline, first session)
## 3. all
## Each participant has both FU and BL MRI, and also the MRIs are of different shape
SUBSET = "all"
assert SUBSET in ["all","BL", "FU"]

# FRACTION
## Determine what fraction [0,1] of participant data to model (for time constraints, may want to do less)
## '1' is all, '0' is none, '0.5' is half
FRACTION = 1

# LEARNING RATES
## 1. 1e-2 #https://www.sciencedirect.com/science/article/pii/S1077314217300620
## 1. 1e-5 #Default
## 1. 1e-3 or 1e-4 https://www.sciencedirect.com/science/article/pii/S1361841516301839, https://www.sciencedirect.com/science/article/pii/S0895611119300771
LEARNING_RATES = [1e-5, 1e-3]

# OPTIMIZERS
## 1. Adam
## 2. SGD #https://www.sciencedirect.com/science/article/pii/S1077314217300620
OPTIMIZERS = ["Adam", "SGD"]

# LOSS FUNCTIONS
LOSS_FUNCTIONS = ["CrossEntropyLoss"]


#CROSS VALIDATION 'K'
## CV = 0, simply doesn't perform cross validation and goes to default hyperparameters
CV = 3

# Hyper-parameters
EPOCHS = 1 # Only relevent for the pytorch model

# Returns the evaluation metric when running the above settings
if run_experiment:
    # Initializing the Modelling Class with Above Parameters
    evaluator = run_tests(epochs=EPOCHS, task=TASK, model_type = MODEL_TYPE, pytorch_version = PYTORCH_VERSION,
                       learning_rates = LEARNING_RATES, optimizers = OPTIMIZERS, loss_functions = LOSS_FUNCTIONS, cv=CV, subset=SUBSET, fraction=FRACTION, penalty = SPACENET_PENALTY
                       , pretrained_resnet = PRETRAINED_RESNET)
    score = evaluator.evaluate()

print("c'est fini")


