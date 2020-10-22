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

class nifty_file:
    def __init__(self):
        self.sub = None
        self.filenameBL = None
        self.pathBL = None
        self.filenameFU = None
        self.pathFU = None
        self.participant_info = None

    def set_participant_info(self, row):
        self.participant_info = row


class train_monai:
    def __init__(self, epochs=2):
        self.train_files = None
        self.val_files = None
        self.train_transforms = None
        self.val_transforms = None
        self.train_loader = None
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.val_loader = None
        self.train_ds = None
        self.val_ds = None
        self.epochs = epochs
        self.file_structure = dict()
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        self.participant_data = pd.read_csv("participants.tsv", sep='\t')
        self.organize_directory()

    # Returns the X and Y of model
    def model_input(self):
        images = []
        labels = []
        nfs = list(self.file_structure.keys())
        random.Random(4).shuffle(nfs) #Same result every time
        for x in nfs:
            nf = self.file_structure[x]
            images.append(nf.pathFU)
            labels.append(1 if list(self.participant_data[self.participant_data['participant_id'] == int(nf.sub)]['group'])[0] == "CB" else 0)
            images.append(nf.pathBL)
            labels.append(1 if list(self.participant_data[self.participant_data['participant_id'] == int(nf.sub)]['group'])[0] == "CB" else 0)

        images = np.array(images)
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

    def setup(self):
        images, labels = self.model_input()
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, shuffle=False)
        print("IMAGES: ", images)
        print("LABELS: ", labels)

        self.train_files = [{"img": img, "label": label} for img, label in zip(X_train, y_train)]
        self.val_files = [{"img": img, "label": label} for img, label in zip(X_test, y_test)]

        # Define transforms for image
        self.train_transforms = Compose(
            [
                LoadNiftid(keys=["img"]),
                AddChanneld(keys=["img"]),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(96, 96, 96)),
                RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
                ToTensord(keys=["img"]),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadNiftid(keys=["img"]),
                AddChanneld(keys=["img"]),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(96, 96, 96)),
                ToTensord(keys=["img"]),
            ]
        )
        # Define dataset, data loader
        check_ds = monai.data.Dataset(data=self.train_files, transform=self.train_transforms)
        check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"])

        # create a training data loader
        self.train_ds = monai.data.Dataset(data=self.train_files, transform=self.train_transforms)
        self.train_loader = DataLoader(self.train_ds, batch_size=2, shuffle=True, num_workers=4,
                                       pin_memory=torch.cuda.is_available())

        # create a validation data loader
        self.val_ds = monai.data.Dataset(data=self.val_files, transform=self.val_transforms)
        self.val_loader = DataLoader(self.val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE: ", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(
            self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-5)

    def train(self):

        # start a typical PyTorch training
        val_interval = 5
        best_metric = -1
        best_metric_epoch = -1
        writer = SummaryWriter()
        for epoch in range(self.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0
            for batch_data in self.train_loader:
                step += 1
                inputs, labels = batch_data["img"].to(self.device), batch_data["label"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(self.train_ds) // self.train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
                    y = torch.tensor([], dtype=torch.long, device=self.device)
                    for val_data in self.val_loader:
                        val_images, val_labels = val_data["img"].to(self.device), val_data["label"].to(self.device)
                        y_pred = torch.cat([y_pred, self.model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                    if acc_metric > best_metric:
                        best_metric = acc_metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(), "best_metric_model_classification3d_dict.pth")
                        print("saved new best metric model")
                    print(
                        "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                            epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()

    def eval(self):
        print("Evaluating...")
        self.model.load_state_dict(torch.load("best_metric_model_classification3d_dict.pth"))
        self.model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            saver = CSVSaver(output_dir="./output")
            real = []
            predicted = []
            for val_data in self.val_loader:
                val_images, val_labels = val_data["img"].to(self.device), val_data["label"].to(self.device)
                val_outputs = self.model(val_images).argmax(dim=1)
                real.append(val_labels.numpy())
                predicted.append(val_outputs.numpy())
                value = torch.eq(val_outputs, val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
                saver.save_batch(val_outputs, val_data["img_meta_dict"])
            flat_real = self.flatten_list(real)
            flat_predicted = self.flatten_list(predicted)
            self.binary_classification(flat_real, flat_predicted)
            metric = num_correct / metric_count
            print("evaluation metric:", metric)
            saver.finalize()


            return metric

    def flatten_list(self, x):
        y = []
        for i in x:
            for j in i:
                y.append(j)
        return y

    def visualize(self):

        # Load image
        bg_img = nibabel.load(('sub-320/ses-BL/anat/sub-320_ses-BL_T1w.nii.gz'))
        bg = bg_img.get_data()
        # Keep values over 4000 as activation map
        act = bg.copy()
        act[act < 6000] = 0.

        # Display the background
        plt.imshow(bg[..., 10].T, origin='lower', interpolation='nearest', cmap='gray')
        # Mask background values of activation map
        masked_act = np.ma.masked_equal(act, 0.)
        plt.imshow(masked_act[..., 10].T, origin='lower', interpolation='nearest', cmap='hot')
        # Cosmetics: disable axis
        plt.axis('off')
        plt.show()
        # Save the activation map
        #nibabel.save(nibabel.Nifti1Image(act, bg_img.get_affine()), 'activation.nii.gz')

    def binary_classification(self, y_true, y_pred):
        print("Classification Results: ")
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("TN", tn, "FP", fp, "FN", fn, "TP", tp)

        # y_pred_class = y_pred_pos > threshold
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        # false_positive_rate = fp / (fp + tn)
#
# def model_input(self, task, sub="func/"):
#        images = []
#        labels = []
#        for y in self.file_structure:
#            task_file = y.replace("/","") + task + "_events.tsv"
#            accuracy = self.events_accuracy(y.replace("/", "", 1), "func/", task_file)
#            label = 1 if accuracy > 0.95 else 0
#            labels.append(label)
#            if "func" in sub:
#                brain_file = y.replace("/","", 1) + "func/" + y.replace("/","") + task + "_bold.nii.gz"
#            else:
#                brain_file = y.replace("/", "", 1) + "anat/" + y.replace("/", "") + "_T1w.nii.gz"
#            images.append(brain_file)
#
#  #        images = np.array(images)
#  #        return images, labels
#
# def events_accuracy(self, folder, subfolder, filename):
#     file = pd.read_csv(folder + subfolder + filename, sep='\t')
#     accuracy = np.mean(file['accuracy'].astype('float'))
#     return accuracy
#
#
#     def get_folders(self):
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         folders = [x[0].replace(dir_path,'') for x in os.walk(dir_path)]
#         for x in folders:
#             if "anat" in x:
#                 for (dirpath, dirnames, filenames) in walk(dir_path + x):
#                     if x.replace("anat", '') in self.file_structure:
#                         self.file_structure[x.replace("anat", '')].append(filenames)
#                     else:
#                         self.file_structure[x.replace("anat", '')] = [filenames]
#             if "func" in x:
#                 for (dirpath, dirnames, filenames) in walk(dir_path + x):
#                     if x.replace("func", '') in self.file_structure:
#                         self.file_structure[x.replace("func", '')].append(filenames)
#                     else:
#                         self.file_structure[x.replace("func", '')] = [filenames]
#         print(self.file_structure.keys())
#         return self.file_structure
