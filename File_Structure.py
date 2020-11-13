import pandas as pd
import random
import os
import numpy as np
from nifty_file import nifty_file


class File_Structure:
    def __init__(self, task):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.file_structure = dict()
        self.task = task
        self.participant_data = pd.read_csv("participants.tsv", sep='\t')

    # Returns the X and Y of model
    def model_input(self, subset, fraction=1):
        images = []
        labels = []
        nfs = list(self.file_structure.keys())
        random.Random(8).shuffle(nfs)  # Shuffle USED TO BE 4, but didn't seem legit
        if self.task == "classification" or self.task == "segmentation":
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
        print(labels)
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
