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

tm = train_monai(epochs=50)
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
    # 10 EPOCHS:
#         YPRED:  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
#         Classification Results:
#         TN 5 FP 1 FN 10 TP 1
#         evaluation metric: 0.35294117647058826
    # 25 EPOCHS:
#         YPRED:  [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
#         Classification Results:
#         TN 3 FP 3 FN 5 TP 6
#         evaluation metric: 0.5294117647058824
    # 50 EPOCHS:
#         Evaluating...
#         YPRED:  [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
#         Classification Results:
#         TN 4 FP 2 FN 6 TP 5
#         evaluation metric: 0.5294117647058824
    # 90 EPOCHS:
#         Evaluating...
#         YPRED:  [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
#         Classification Results:
#         TN 4 FP 2 FN 5 TP 6
#         evaluation metric: 0.5882352941176471



# 2. Regression

# 3. Segmentation

