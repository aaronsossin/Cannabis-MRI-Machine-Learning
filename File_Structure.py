from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadNiftid, RandRotate90d, Resized, ScaleIntensityd, ToTensord
from sklearn.model_selection import train_test_split
from glob import glob
import random
import nibabel
from matplotlib import pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import CSVSaver
from monai.transforms import AddChanneld, Compose, LoadNiftid, Resized, ScaleIntensityd, ToTensord
from sklearn.metrics import confusion_matrix
from nilearn.decoding import SpaceNetRegressor
from nilearn.image import smooth_img, resample_img, load_img, index_img, concat_imgs
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_stat_map
from nilearn import plotting
from nilearn.plotting import show
import nilearn
from sklearn.metrics import r2_score
from nilearn.image import mean_img
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from nifty_file import nifty_file

class File_Structure:
    def __init__(self, task):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.file_structure = dict()
        self.task = task
        self.participant_data = pd.read_csv("participants.tsv", sep='\t')

    # Returns the X and Y of model
    def model_input(self, subset, fraction = 1):
        images = []
        labels = []
        nfs = list(self.file_structure.keys())
        random.Random(5).shuffle(nfs)  # Shuffle USED TO BE 4, but didn't seem legit
        if self.task == "classification":
            for x in nfs:
                nf = self.file_structure[x]
                if subset == "all" or subset == "FU":
                    images.append(nf.pathFU)
                    labels.append(1 if list(
                        self.participant_data[self.participant_data['participant_id'] == int(nf.sub)]['group'])[
                                           0] == "CB" else 0)
                if subset == "all" or subset == "BL":
                    images.append(nf.pathBL)
                    labels.append(1 if list(
                        self.participant_data[self.participant_data['participant_id'] == int(nf.sub)]['group'])[
                                           0] == "CB" else 0)
        elif self.task == "regression":
            print("CUDIT TASK")
            for x in nfs:
                nf = self.file_structure[x]
                if subset == "all" or subset == "FU":
                    images.append(nf.pathFU)
                    labels.append(list(
                        self.participant_data[self.participant_data['participant_id'] == int(nf.sub)][
                            'cudit-total-follow-up'].astype('float'))[0])
                if subset == "all" or subset == "BL":
                    images.append(nf.pathBL)
                    labels.append(list(
                        self.participant_data[self.participant_data['participant_id'] == int(nf.sub)][
                            'cudit-total-baseline'].astype('float'))[0])
        images = np.array(images)
        lim = int(fraction * len(images))
        images = images[0:lim]
        labels = labels[0:lim]
        return images, labels

    def organize_directory(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        folders = [x[2] for x in os.walk(dir_path)]
        brain_files = [x[0] for x in folders if ".nii.gz" in str(x)]
        for y in brain_files:
            split = y.split('_')
            sub = split[0].split('-')[1]
            if sub in self.file_structure:
                nf = self.file_structure[sub]
            else:
                nf = nifty_file()
                nf.set_participant_info(self.participant_data[self.participant_data['participant_id'] == str(sub)])
                nf.sub = sub
                self.file_structure[sub] = nf

            type = split[1]

            if type == "ses-BL":
                nf.filenameBL = y
                nf.pathBL = str("sub-" + sub + "/" + type + "/anat/" + y)
            elif type == "ses-FU":
                nf.filenameFU = y
                nf.pathFU = str("sub-" + sub + "/" + type + "/anat/" + y)
        return self.file_structure
