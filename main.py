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

from train_monai import train_monai
from plot_participant_data import plot_participant_data

from nilearn import datasets, image

##########################################
"""
Environment set up: (i think this is it)
1. pip install nilearn
2. pip install monai
3. pip install torch
"""
##############################################


#Only run_model need ever be run.
run_model = True
plot_participants = False
test_epochs = False

#IDEAS
# Try Resnet121/101
# N1Loss looks good
# TRANSFER LEARNING WITH MONAI saved dict!!!!!!!!!!!!!!!!!!!!! Or, maybe a new unsupervised?

# Exclamation
print("BITCH")

# TASKS:
## 1. 'classification'
## 2. 'regression'
## 3. 'segmentation'
task = "classification"

# MODELS:
## 1. 'monai'
## 2. 'nilearn'
## MORE TO COME
model_type = "monai"

# MONAI DENSENET VERSION
## 1. 121
## 2. 201
## 3. 169
## 4. 264
## 5. 1 (AlexNet)
pytorch_version = 1

"""
Note: The 'nilearn' regression also performs segmentation, but different process than the explicit segmenation
"""

# SUBSET:
## 1. FU (only follow up scores (after the 3 years)
## 2. BL (baseline, first session)
## Each participant has both FU and BL MRI, and also the MRIs are of different shape
subset = "all"

# FRACTION
## Determine what fraction [0,1] of participant data to model (for time constraints, may want to do less)
## '1' is all, '0' is none, '0.5' is half
fraction = 1

# Hyper-parameters
epochs = 6 # Only relevent for the MONAI model
penalty = "graph-net" #Only  for ni-learn, either "graph-net" or "tv-l1"

# Columns to plot participant data
to_plot = ['cudit total baseline', 'cudit total follow-up',
    'audit total baseline',	'audit total follow-up', 'age at baseline ',
    'age at onset first CB use',	'age at onset frequent CB use']

# Initializing the Modelling Class with Abote parameters
tm = train_monai(epochs=epochs, task=task, model_type = model_type, pytorch_version = pytorch_version)

# Returns the evaluation metric when running the above settings
if run_model:
    score = tm.evaluate(subset=subset, fraction=fraction, kernel='linear', penalty=penalty)

if plot_participants:
    pd = plot_participant_data(pd.read_csv("participants.tsv", sep='\t'), to_plot)
    pd.plot_cats()

if test_epochs:
    for x in epochs:
        tm = train_monai(epochs = x, task = task, model_type = model_type)
        score = tm.evaluate(subset='FU')
        print(x, ": ", score)


# Results - Don't delete

#Standard = 4 epochs, shuffled, densenet121 from monai, cross entropy loss, Adam1e-5,
# 1. Controls vs. Heavy Users
    # DENSENET 121
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

    # DENSENET 264
    # 10 EPOCHS
    #     Evaluating...
        # Classification Results:
        # TN 6 FP 0 FN 8 TP 3
        # evaluation metric: 0.5294117647058824

    # 25 EPOCHS
        #     Evaluating...
        # Classification Results:
        # [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
        # TN 3 FP 3 FN 6 TP 5
        # evaluation metric: 0.47058823529411764

    # 50 EOPCHS

            #         Classification Results:
            # [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
            # TN 6 FP 0 FN 8 TP 3
            # evaluation metric: 0.5294117647058824
   # 169
    # 10
        #     Classification Results:
        # [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        # TN 6 FP 0 FN 8 TP 3
        # evaluation metric: 0.5294117647058824
    # 25
        #         Evaluating...
        # Classification Results:
        # [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
        # TN 6 FP 0 FN 7 TP 4
        # evaluation metric: 0.5882352941176471
     # 50
        #     Evaluating...
        # Classification Results:
        # [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1]
        # TN 3 FP 3 FN 4 TP 7
        # evaluation metric: 0.5882352941176471
    # SEGRESNET

    # SEGRESNETVAE AUTOENCODER USES UNPSERVISED LARNING



# fig = plt.figure()
# plt.plot([10, 25, 50, 90], [0.353, 0.530, 0.530, 0.588], label="DenseNet121")
# plt.plot([10, 25, 50, 90], [0.530, 0.470, 0.530, 0.530], label="DenseNet264")
# plt.plot([10, 25, 50, 90], [0.530, 0.588, 0.588, 0.611], label="DenseNet169")
# plt.plot([0,90], [0.5, 0.5], '--')
# plt.title("Baseline Performances of MONAI-defined DenseNet Architectures")
# plt.xlabel("Epoch Size")
# plt.ylabel("Accuracy (%)")
# plt.legend()
# plt.show()

# 2. Regression

# 3. Segmentation

