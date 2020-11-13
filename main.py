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


#To Run Model set to True
run_experiment = True

#IDEAS
# TRANSFER LEARNING WITH MONAI saved dict!!!!!!!!!!!!!!!!!!!!! Or, maybe a new unsupervised?

#To do
# Do SVM regression

# Exclamation
print("~...Beep Boop Beep...~")

# TASKS:
## 1. 'classification'
## 2. 'regression'
## 3. 'segmentation'
TASK = "regression"

# MODELS:
## 1. 'PyTorch'
## 2. 'SpaceNet'
## 3. 'SVM'
## MORE TO COME
MODEL_TYPE = "SVM"

SPACENET_PENALTY = "tv-l1" #Only  for ni-learn, either "graph-net" or "tv-l1"

# PYTORCH MODELS
## 1. 121
## 2. 201
## 3. 169
## 4. 264
## 5. 1 ~(AlexNet)
## 6. 2 ~(ResNet)
PYTORCH_VERSION = 2

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
#optimizers = [torch.optim.Adam, torch.optim.SGD]
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

if False:
    # Columns to plot participant data
    to_plot = ['cudit total baseline', 'cudit total follow-up',
    'audit total baseline',	'audit total follow-up', 'age at baseline ',
    'age at onset first CB use',	'age at onset frequent CB use']

    pd = plot_participant_data(pd.read_csv("participants.tsv", sep='\t'), to_plot)
    pd.plot_cats()

print("c'est fini")

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


