import monai
import glob
import shutil
import tempfile

from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import numpy as np
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

dir_path = os.path.dirname(os.path.realpath(__file__))

tm = train_monai(epochs=2)

run_model = True

if run_model:
    tm.setup()
    tm.train()
    tm.eval()

