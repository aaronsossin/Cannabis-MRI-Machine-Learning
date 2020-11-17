import os
import json
import monai
import nilearn
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from monai.data import CSVSaver
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadNiftid, Resized, ScaleIntensityd, ToTensord
from monai.transforms import RandRotate90d
from nilearn import plotting
from nilearn.decoding import SpaceNetRegressor, SpaceNetClassifier
from nilearn.image import smooth_img, resample_img, load_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from nilearn.plotting import show
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from AlexNet import AlexNet3D
from File_Structure import File_Structure
from sklearn.model_selection import KFold
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template
from nilearn import datasets, image
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import resnet

"""
This class is responsible for all the experiments in the report
Depending on the inputted parameters, the 'evaluate()' method will call on the correct methods
To understand this class, it is best to look at the evaluate() function and see the methods which it calls

Functionality Includes:
    - Decoders (SpaceNet/SVM)
    - All PyTorch Implementations

Methods in order
    - __init__() : initialize parameters and organize file structure internally upon creation
    - evaluate : depending on which model/task, calls appropriate methods
    - pytorch_cv_grid_search: Runs Grid Search Cross Validation on the PyTorch models
    - setup : initializes all the variables needed to run PyTorch models 
        like Resampling Images, DataLoaders, Model Selection, etc...
    - pytorch_train 
    = pytorch_eval
    - SVM
    - SpaceNet
    - Helper Functions Below
"""

class run_tests:
    def __init__(self, epochs=2, task='control', model_type="nilearn", pytorch_version=0, loss_functions=[],
                 optimizers=[], learning_rates=[], cv=3, subset="all", fraction=1, penalty='tv-l1',
                 pretrained_resnet=False):
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
        self.shape = None
        self.output_shape = None

        self.file_structure = dict()
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        # Based on Parameters
        self.pretrained_resnet = pretrained_resnet
        self.penalty = penalty
        self.fraction = fraction
        self.subset = subset
        self.loss_functions = loss_functions
        self.optimizers = optimizers
        self.learning_rates = learning_rates
        self.cv = cv
        self.task = task
        self.model_type = model_type
        self.epochs = epochs
        self.pytorch_version = pytorch_version
        self.participant_data = pd.read_csv("participants.tsv", sep='\t')

        # Runs on Initialization to Create File Structure
        self.saved_model_dict = "best_metric_" + self.task + ":" + self.model_type + ":" + str(
            self.pytorch_version) + ".pth"
        self.File_Structure = File_Structure(self.task)
        self.File_Structure.organize_directory()

        if self.task == "classification":
            self.output_shape = 2
        else:
            self.output_shape = 30

    def evaluate(self):

        score = 0  # Evaluation score - depends on task

        # shape = the shape of MRI. Since BL and FU have different shapes, must standardize when using nilearn
        if self.subset == "all":
            self.shape = (256, 256, 256)
        elif self.subset == "FU":
            self.shape = (256, 256, 170)
        else:
            self.shape = (256, 182, 256)

        # images : MRI Image FileNames, Labels: Associated label depending on regression/classification
        images, labels = self.File_Structure.model_input(self.subset, self.fraction)

        X = np.array(images)
        y = np.array(labels)

        # SpaceNet Decoder
        if self.model_type == "SpaceNet":

            score = self.SpaceNet(X, y)

        # SVM Decoder
        elif self.model_type == "SVM":

            score = self.SVM(X, y)

        elif self.model_type == "PyTorch":

            # Cross Validation
            if self.cv > 0:
                self.pytorch_cv_grid_search(X, y)

            # Cross Validation Not Applied (Controls, Preliminary Experiments)
            X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, shuffle=False)

            # Setting up data in Tensors to be trained on
            self.setup(X_train, X_test, y_train, y_test, 1e-5, "SGD", "CrossEntropyLoss")

            # Train Model
            self.pytorch_train()

            # Evaluate Model
            score = self.pytorch_eval()

        print("Score: ", score)

    def pytorch_cv_grid_search(self, X, y):

        kf = KFold(n_splits=self.cv, random_state=None, shuffle=False)
        counter = 0

        scores = dict()  # Tracking Scores of each CV Run

        for train_index, test_index in kf.split(X):

            scores[counter] = dict()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for optimizer in self.optimizers:

                for learning_rate in self.learning_rates:

                    for loss_function in self.loss_functions:
                        key = str(optimizer) + ":" + " LR: " + str(learning_rate) + ":" + str(loss_function)
                        print(key)

                        self.setup(X_train, X_test, y_train, y_test, learning_rate, optimizer, loss_function)
                        self.pytorch_train()
                        score = self.pytorch_eval()

                        scores[counter][key] = score
                        print(scores)

            counter += 1

        # Saving CV Output to a File
        stringified_dict = json.loads(json.dumps(scores), parse_int=str)
        print(stringified_dict)
        outF = open(self.saved_model_dict.replace('.pth', '.txt'), "w")
        outF.write(str(stringified_dict))
        outF.close()

    def setup(self, X_train, X_test, y_train, y_test, learning_rate, optimizer, loss_function):

        # Extract images and labels from the participant files in directory

        self.train_files = [{"img": img, "label": label} for img, label in zip(X_train, y_train)]
        self.val_files = [{"img": img, "label": label} for img, label in zip(X_test, y_test)]

        # Target (_,_,_) Resampling Shape

        if self.pytorch_version == 1:  # AlexNet Requires this Size
            s_size = 227
        else:
            s_size = 96

        # Define transforms for image

        self.train_transforms = Compose(
            [
                LoadNiftid(keys=["img"]),
                AddChanneld(keys=["img"]),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(s_size, s_size, s_size)),
                RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
                ToTensord(keys=["img"]),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadNiftid(keys=["img"]),
                AddChanneld(keys=["img"]),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(s_size, s_size, s_size)),
                ToTensord(keys=["img"]),
            ]
        )

        # Dataset and Dataloader for PyTorch

        check_ds = monai.data.Dataset(data=self.train_files, transform=self.train_transforms)
        check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"])

        # Training DataLoader

        self.train_ds = monai.data.Dataset(data=self.train_files, transform=self.train_transforms)
        self.train_loader = DataLoader(self.train_ds, batch_size=2, shuffle=True, num_workers=4,
                                       pin_memory=torch.cuda.is_available())

        # Testing DataLoader

        self.val_ds = monai.data.Dataset(data=self.val_files, transform=self.val_transforms)
        self.val_loader = DataLoader(self.val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

        # Using CUDA if Available

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE: ", "cuda" if torch.cuda.is_available() else "cpu")

        # Model Selection based on Input Parameters
        # Output shape dependent on regression/classification
        if self.pytorch_version == 1:
            self.model = AlexNet3D()
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.pytorch_version == 2:
            self.model = resnet.resnet_101(pretrained=self.pretrained_resnet, progress=True)
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.pytorch_version == 121:
            self.model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1,
                                                                  out_channels=self.output_shape).to(
                self.device)
        elif self.pytorch_version == 169:
            self.model = monai.networks.nets.densenet.densenet264(spatial_dims=3, in_channels=1,
                                                                  out_channels=self.output_shape).to(self.device)
        elif self.pytorch_version == 201:
            self.model = monai.networks.nets.densenet.densenet169(spatial_dims=3, in_channels=1,
                                                                  out_channels=self.output_shape).to(self.device)
        elif self.pytorch_version == 264:
            self.model = monai.networks.nets.densenet.densenet201(spatial_dims=3, in_channels=1,
                                                                  out_channels=self.output_shape).to(self.device)

        # Grid Search Parameters: Optimizer, Loss Function and Learning Rate

        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        if self.task == "regression":
            self.loss_function = torch.nn.MSELoss()
        else:
            if loss_function == "CrossEntropyLoss":
                self.loss_function = torch.nn.CrossEntropyLoss()
            elif loss_function == "MSELoss":
                self.loss_function = torch.nn.MSELoss()
            elif loss_function == "NLLLoss":
                self.loss_function = torch.nn.NLLLoss()
        return [X_train, X_test, y_train, y_test]

    """
    Trains all PyTorch models including MONAI and ALexNet/ResNet
    
    Periodically outputs progress for later plotting
    
    Depending on classification or regression task, has slight alterations to its mechanics
    
    Saves trained model weights to a "_.pth" file
    """
    def pytorch_train(self):
        acc_scores = dict()
        auc_scores = dict()

        # start a typical PyTorch training
        val_interval = 5
        best_metric = -1 if self.task == "classification" else 1e8
        best_metric_epoch = -1
        writer = SummaryWriter()
        torch.save(self.model.state_dict(), self.saved_model_dict)  # ADDED FOR SMALL EPOCH STUFF
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
                if self.pytorch_version == 1:
                    outputs = torch.nn.functional.softmax(outputs, dim=0)
                if self.task == "classification":
                    loss = self.loss_function(outputs, labels)
                else:
                    loss = self.loss_function(outputs, labels.view(-1, 1).float())

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(self.train_ds) // self.train_loader.batch_size
                if step % 3 == 0:
                    print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
                    y = torch.tensor([], dtype=torch.long, device=self.device)
                    real = []
                    predicted = []
                    for val_data in self.val_loader:
                        val_images, val_labels = val_data["img"].to(self.device), val_data["label"].to(self.device)
                        y_pred = torch.cat([y_pred, self.model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)
                        if self.task == "regression":
                            real.append(val_labels.cpu().numpy())
                            predicted.append(self.model(val_images).argmax(dim=1).cpu().numpy())

                    if self.task == "classification":
                        acc_value = torch.eq(y_pred.argmax(dim=1), y)
                        acc_metric = acc_value.sum().item() / len(acc_value)
                        auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                    else:
                        acc_metric = mean_squared_error(self.flatten_list(real), self.flatten_list(predicted))
                        auc_metric = 0
                    acc_scores[epoch] = acc_metric
                    auc_scores[epoch] = auc_metric
                    if (acc_metric >= best_metric and self.task == "classification") or (
                            acc_metric <= best_metric and self.task == "regression"):
                        best_metric = acc_metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(), self.saved_model_dict)
                    print(
                        "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                            epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()
        print(acc_scores, "\n", auc_scores)



    def pytorch_eval(self):

        print("Evaluating...")
        self.model.load_state_dict(torch.load(self.saved_model_dict))
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
                real.append(val_labels.cpu().numpy())
                predicted.append(val_outputs.cpu().numpy())
                value = torch.eq(val_outputs, val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
                saver.save_batch(val_outputs, val_data["img_meta_dict"])
            flat_real = self.flatten_list(real)
            flat_predicted = self.flatten_list(predicted)
            if self.task == "classification":
                score = self.binary_classification(flat_real, flat_predicted)
                metric = num_correct / metric_count
            else:
                print("REAL: ", flat_real)
                print("PRED: ", flat_predicted)
                score = mean_squared_error(flat_real, flat_predicted)
                print(score)
            saver.finalize()

            return score

    def SVM(self, X, y):  # (-10.0, -26.0, 28.0) COORDINATES WERE FOUND
        print("Resampling..")

        X_resampled = self.nilearn_resample(X)

        masker = NiftiMasker(smoothing_fwhm=1,
                             standardize=True, memory="nilearn_cache", memory_level=1)

        X = masker.fit_transform(X_resampled)
        mean_img = nilearn.image.mean_img(X_resampled)
        # Model
        print("Training...")
        feature_selection = SelectPercentile(f_classif, percentile=0.5)

        svc = SVC(kernel='linear')
        anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

        anova_svc.fit(X, y)
        y_pred = anova_svc.predict(X)
        # anova_svc.fit(X, y)
        # Compute the prediction accuracy for the different folds (i.e. session)
        if self.task == "classification":
            scoring = "accuracy"
        else:
            scoring = "neg_mean_absolute_error"
        # cv_scores = cross_val_score(anova_svc, X, y, cv = 1, scoring=scoring)
        # print(cv_scores)

        # Return the corresponding mean prediction accuracy
        # score = cv_scores.mean()

        # print("Score at task: ", score)
        from sklearn.metrics import mean_squared_error
        # Only half of data
        # anova_svc.fit(X, y)
        # y_pred = anova_svc.predict(X)
        if self.task == "classification":
            output, stats = self.binary_classification(y, y_pred)
            print(stats)
        else:
            stats = mean_squared_error(y, y_pred)
            print(stats)
        coef = svc.coef_  # was svc
        print("COEF\n", coef)
        # reverse feature selection
        coef = feature_selection.inverse_transform(coef)
        print("COEF_INVERSE\n", coef)
        # reverse masking
        weight_img = masker.inverse_transform(coef)

        # Use the mean image as a background to avoid relying on anatomical data

        # Create the figure
        plot_stat_map(weight_img, mean_img, title='SVM weights')

        show()

        from nilearn import datasets, image
        niimg = datasets.load_mni152_template()
        # Find the MNI coordinates of the voxel (50, 50, 50)
        l = image.coord_transform(70, 21, -20, niimg.affine)
        print("MAYBE?", l)

        return stats

    """
    Nilearn defined SpaceNet Decoder: Can do Classification or Regression
    """
    def SpaceNet(self, X, y):

        background_computed = False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        X_test = self.nilearn_resample(X_test)
        X_train = self.nilearn_resample(X_train)
        X_all = np.append(X_train, X_test)
        if not background_computed:
            background_img = nilearn.image.mean_img(X_all)
            print("background computer")
            background_computed = True
        scores = []
        # Predictions on test set
        if self.task == "classification":
            decoder = self.new_SpaceNet(self.task, self.penalty)
            decoder.fit(X_train, y_train)
            print("decoder fit")
            y_pred = np.round(decoder.predict(X_test).ravel())
            accuracy = np.average([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))])
            print(accuracy)
            scores.append(accuracy)
        else:
            decoder = self.new_SpaceNet(self.task, self.penalty)
            decoder.fit(X_train, y_train)
            print("decoder fit")
            y_pred = decoder.predict(X_test).ravel()
            mse = np.mean(np.abs(y_test - y_pred))
            score = mse
            r2 = r2_score(y_test, y_pred)
            print(mse)
            scores.append(mse)
        score = np.average(scores)
        print(score)

        coef_img = decoder.coef_img_

        suptitle = "SCORE: " + str(score)

        # Plot
        plt.figure()
        plt.suptitle(suptitle)
        linewidth = 3
        ax1 = plt.subplot('211')
        ax1.plot(y_test, label="True Cudit Score", linewidth=linewidth)
        ax1.plot(y_pred, '--', c="g", label="Predicted Cudit Score", linewidth=linewidth)
        ax1.set_ylabel("Cudit Score")
        plt.legend(loc="best")
        ax2 = plt.subplot("212")
        ax2.plot(y_test - y_pred, label="True - predicted",
                 linewidth=linewidth)
        ax2.set_xlabel("subject")
        plt.legend(loc="best")

        title = self.penalty + " weights"
        plot_stat_map(coef_img, background_img, title=title)

        plt.show()

        return score

    "Computes some Useful Stats for Binary Classification Analyses"
    def binary_classification(self, y_true, y_pred):
        print("Classification Results: ")
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        output = "TN:" + str(tn) + "FP:" + str(fp) + "FN:" + str(fn) + "TP:" + str(tp)
        stats = dict()
        stats["sensitivity"] = tp / (tp + fn)
        stats["specificity"] = tn / (fp + tn)
        stats["balanced_accuracy"] = (stats["sensitivity"] + stats["specificity"]) / 2.0
        stats["accuracy"] = (tp + tn) / (tp + tn + fp + fn)

        print(output)
        print(stats)
        return [output, stats]

    """
    Convert coordinates to the MNI Template for Referencing
    """
    def convert_coordinates(self, x, y, z):
        niimg = datasets.load_mni152_template()
        converted = image.coord_transform((x, y, z), niimg.affine)
        print("CONVERTED COORDINATES: ", converted)

    """
    Creates a new SpaceNet Decoder
    """
    def new_SpaceNet(self, type, loss_function):
        if type == "classification":
            decoder = SpaceNetClassifier(memory="nilearn_cache", penalty=loss_function,
                                         screening_percentile=5., memory_level=1)
        else:
            decoder = SpaceNetRegressor(memory="nilearn_cache", penalty=loss_function,
                                        screening_percentile=5., memory_level=1)
        return decoder

    """
    Resampling MRI image to be processed by Decoder
    """
    def nilearn_resample(self, X):
        template = load_mni152_template()
        resampled_X = []
        for x in X:
            z = resample_to_img(x, template)
            z_affine = z.affine
            resampled_X.append(resample_img(x, target_affine=z_affine, target_shape=self.shape))
        return resampled_X

    """
    Helper Function to Visualize an MRI File
    """
    def visualize(self, c=None):
        if c != None:
            # plot_stat_map(c)
            # show()
            plotting.plot_glass_brain(c)
            show()
        else:
            # plotting.plot_stat_map('sub-320/ses-FU/anat/sub-320_ses-FU_T1w.nii.gz')
            plotting.plot_stat_map('sub-/ses-FU/anat/sub-133_ses-FU_T1w.nii.gz')
            show()

    """
    Auxiliary Helper Function to Flatten a List
    """
    def flatten_list(self, x):
        y = []
        for i in x:
            for j in i:
                y.append(j)
        return y

