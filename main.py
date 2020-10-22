import numpy as np
import monai
import glob
import shutil
import tempfile

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

from train_monai import train_monai
from plot_participant_data import plot_participant_data

run_model = True
plot_participants = False
test_epochs = False
epochs = [2,3,4]

# Try Resnet121/101


to_plot = ['cudit total baseline', 'cudit total follow-up',
    'audit total baseline',	'audit total follow-up', 'age at baseline ',
    'age at onset first CB use',	'age at onset frequent CB use']

tm = train_monai(epochs=1)
pd = plot_participant_data(pd.read_csv("participants.tsv", sep='\t'), to_plot)

#tm.visualize()

if plot_participants:
    pd.plot_cats()

if run_model:
    tm.setup()
    tm.train()
    score = tm.eval()

if test_epochs:
    for x in epochs:
        tm.setup()
        tm.train()
        score = tm.eval
        print(x, ": ", score)


# Results

#Standard = 4 epochs, shuffled, densenet121 from monai, cross entropy loss, Adam1e-5,
# 1. Controls vs. Heavy Users
    # 4 Epochs, shuffled, standard featuers, guesses 1 everytime.
    # ...
    # epoch 92 average loss: 0.3795
#current epoch: 92 current accuracy: 0.5882 current AUC: 0.5286 best accuracy: 0.6471 at epoch 2

# 2. Regression

# 3. Segmentation

