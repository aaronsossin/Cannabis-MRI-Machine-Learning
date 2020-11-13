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

import resnet

class run_tests:
    def __init__(self, epochs=2, task='control', model_type= "nilearn", pytorch_version = 0, loss_functions = [],
                 optimizers = [], learning_rates = [], cv=3, subset = "all", fraction = 1, penalty='tv-l1',
                 pretrained_resnet = False):
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
        self.saved_model_dict = "best_metric_" + self.task + ":" + self.model_type + ":" + str(self.pytorch_version) + ".pth"
        self.File_Structure = File_Structure(self.task)
        self.File_Structure.organize_directory()

    def pytorch_cv_grid_search(self, X, y):

        kf = KFold(n_splits = self.cv, random_state=None,shuffle=False)
        counter = 0
        scores = dict()
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

            counter += 1

        stringified_dict = json.loads(json.dumps(scores), parse_int=str)
        print(stringified_dict)
        outF = open(self.saved_model_dict.replace('.pth','.txt'), "w")
        outF.write(str(stringified_dict))
        outF.close()


    def evaluate(self):

        score = 0
        # shape = the shape of MRI. Since BL and FU have different shapes, must standardize when using nilearn
        if self.subset == "all":
            self.shape = (256, 256, 256)
        elif self.subset == "FU":
            self.shape = (256, 256, 170)
        else:
            self.shape = (256, 182, 256)

        # The train/eval data specialized for either regression or classification
        images, labels = self.File_Structure.model_input(self.subset, self.fraction)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, shuffle=False)
        X = np.append(X_train, X_test)
        y = np.append(y_train, y_test)

        # Run model
        if self.model_type == "SpaceNet":

            score = self.SpaceNet(X, y)

        elif self.model_type == "SVM":

            score = self.SVM(X, y)

        elif self.model_type == "PyTorch":

            if self.cv > 0 and self.task == "classification":
                self.pytorch_cv_grid_search(X, y)

            self.setup(X_train, X_test, y_train, y_test, 1e-5, "SGD", "CrossEntropyLoss")

            if self.task == "classification":
                self.pytorch_train()
                score = self.pytorch_eval()

            else:
                print("Regression not available for pytorch")

        print("Score: ", score)

    # Setup the Monai File Structure and Train/Eval images/labels
    def setup(self, X_train, X_test, y_train, y_test, learning_rate, optimizer, loss_function):

        # Extract images and labels from the participant files in directory

        #X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.4, shuffle=False)


        self.train_files = [{"img": img, "label": label} for img, label in zip(X_train, y_train)]
        self.val_files = [{"img": img, "label": label} for img, label in zip(X_test, y_test)]

        # Shape of Image to Resample
        if self.pytorch_version  == 1: #AlexNet
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

        # MODELS:
        if self.pytorch_version == 1:
            self.model = AlexNet3D()
        elif self.pytorch_version == 2:
            self.model = resnet.resnet_101(pretrained=self.pretrained_resnet, progress=True)
        elif self.pytorch_version == 121:
            self.model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(
                self.device)
        elif self.pytorch_version == 169:
            self.model = monai.networks.nets.densenet.densenet264(spatial_dims=3, in_channels=1, out_channels=2).to(self.device)
        elif self.pytorch_version == 201:
            self.model = monai.networks.nets.densenet.densenet169(spatial_dims=3, in_channels=1, out_channels=2).to(self.device)
        elif self.pytorch_version == 264:
            self.model = monai.networks.nets.densenet.densenet201(spatial_dims=3, in_channels=1, out_channels=2).to(self.device)
        #self.model = monai.networks.nets.SegResNetVAE(input_image_size=(96,96,96)).to(self.device)

        ############## GOOD OPTIONS FOR HYPER-PARAMETER TESTING #####################
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        if loss_function == "CrossEntropyLoss":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss_function == "MSELoss":
            self.loss_function = torch.nn.MSELoss()
        elif loss_function == "NLLLoss":
            self.loss_function = torch.nn.NLLLoss()
        return [X_train, X_test, y_train, y_test]

    # Entire Function copy-pasted from ....com, trains the model and saves it to directory
    def pytorch_train(self):
        acc_scores = dict()
        auc_scores = dict()
        # start a typical PyTorch training
        val_interval = 5
        best_metric = -1
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
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(self.train_ds) // self.train_loader.batch_size
                if step % 3 == 0:
                    True
                    #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            #print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
                    y = torch.tensor([], dtype=torch.long, device=self.device)
                    for val_data in self.val_loader:
                        val_images, val_labels = val_data["img"].to(self.device), val_data["label"].to(self.device)
                        y_pred = torch.cat([y_pred, self.model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)
                    # y_pred_numpy = list(np.array(y_pred.argmax(dim=1).cpu()))
                    # y_numpy = list(np.array(y.cpu()))
                    # cm = confusion_matrix(y_numpy, y_pred_numpy)
                    # tn, fp, fn, tp = cm.ravel()
                    # acc_metric = (tp / (tp + fn) + tn / (fp + tn))/2.0 #Balanced Accuracy
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                    acc_scores[epoch] = acc_metric
                    auc_scores[epoch] = auc_metric
                    if acc_metric > best_metric:
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

    # Evaluates best model that was saved
    # Only for classification right now

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
            binary_ = self.binary_classification(flat_real, flat_predicted)
            metric = num_correct / metric_count
            saver.finalize()

            return binary_

    def binary_classification(self, y_true, y_pred):
        print("Classification Results: ")
        print(y_true)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        output = "TN:" + str(tn) + "FP:" + str(fp) + "FN:" + str(fn) + "TP:" + str(tp)
        stats = dict()
        stats["sensitivity"] = tp / (tp + fn)
        stats["specificity"] = tn / (fp + tn)
        stats["balanced_accuracy"] = (stats["sensitivity"] + stats["specificity"])/2.0
        stats["accuracy"] = (tp + tn) / (tp + tn + fp + fn)

        print(output)
        print(stats)
        return [output, stats]

    def SVM(self,  X, y):
        print("Resampling..")
        resampled_X = self.nilearn_resample(X)

        masker = NiftiMasker(smoothing_fwhm=4,
                                standardize=True, memory="nilearn_cache", memory_level=1)

        X = masker.fit_transform(resampled_X)

        # Model
        print("Training...")
        feature_selection = SelectPercentile(f_classif, percentile=5)

        svc = SVC(kernel='linear')
        anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])


        # Compute the prediction accuracy for the different folds (i.e. session)
        if self.task == "classification":
            scoring = "accuracy_score"
        else:
            scoring = "neg_mean_absolute_error"
        cv_scores = cross_val_score(anova_svc, X, y, cv = self.cv, scoring=scoring)
        print(cv_scores)

        # Return the corresponding mean prediction accuracy
        score = cv_scores.mean()

        print("Score at task: ", score)

        #Only half of data
        anova_svc.fit(X, y)
        coef = svc.coef_
        print("COEF\n", coef)
        # reverse feature selection
        coef = feature_selection.inverse_transform(coef)
        print("COEF_INVERSE\n", coef)
        # reverse masking
        weight_img = masker.inverse_transform(coef)

        # Use the mean image as a background to avoid relying on anatomical data
        mean_img = nilearn.image.mean_img(resampled_X)

        # Create the figure
        plot_stat_map(weight_img, mean_img, title='SVM weights')

        show()

        return score


    # Nilearn_Regression

    def nilearn_resample(self, X):
        template = load_mni152_template()
        resampled_X = []
        for x in X:
            z = resample_to_img(x, template)
            z_affine = z.affine
            resampled_X.append(resample_img(x, target_affine=z_affine, target_shape=self.shape))
        return resampled_X

    def new_decoder(self, type, loss_function):
        if type == "classification":
            decoder = SpaceNetClassifier(memory="nilearn_cache", penalty=loss_function,
                                         screening_percentile=5., memory_level=1)
        else:
            decoder = SpaceNetRegressor(memory="nilearn_cache", penalty=loss_function,
                                        screening_percentile=5., memory_level=1)
        return decoder

    def SpaceNet(self,X, y):

        kf = KFold(n_splits=self.cv)
        background_computed = False
        for train_index, test_index in kf.split(X):
            print("New CV Run")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_test = self.nilearn_resample(X_test)
            X_train = self.nilearn_resample(X_train)
            X_all = np.append(X_train, X_test)
            if not background_computed:
                background_img = nilearn.image.mean_img(X_all)
                background_computed = True
            scores = []
            # Predictions on test set
            if self.task == "classification":
                decoder = self.new_decoder(self.task, self.penalty)
                decoder.fit(X_train, y_train)
                print("decoder fit")
                y_pred = np.round(decoder.predict(X_test).ravel())
                accuracy = np.average([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))])
                print(accuracy)
                scores.append(accuracy)
            else:
                decoder = self.new_decoder(self.task, self.penalty)
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

        suptitle= "SCORE: " + str(score)

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

        title = self.penalty + "weights"
        plot_stat_map(coef_img, background_img, title=title)

        plt.show()

        return score

    def visualize(self, c=None):
        if c != None:
            #plot_stat_map(c)
            #show()
            plotting.plot_glass_brain(c)
            show()
        else:
            #plotting.plot_stat_map('sub-320/ses-FU/anat/sub-320_ses-FU_T1w.nii.gz')
            plotting.plot_stat_map('sub-/ses-FU/anat/sub-133_ses-FU_T1w.nii.gz')
            show()

    def flatten_list(self, x):
        y = []
        for i in x:
            for j in i:
                y.append(j)
        return y

    def plot_grid_search(self, cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
        print("doesn't work lmao")


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


#net = NeuralNetClassifier(
        #     self.model,
        #     max_epochs=50,
        #     # Shuffle training data on each epoch
        #     iterator_train__shuffle=True,
        # )
        # # deactivate skorch-internal train-valid split and verbose logging
        # net.set_params(train_split=False, verbose=0)
        # params = {
        #     'lr': self.learning_rates,
        #     'criterion': ['Adam', 'SGD']
        # }
        # gs = GridSearchCV(net, params, refit=False, cv=3, scoring='balanced_accuracy', verbose=2)
        #
        # gs.fit(X, y)
        # print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
        #
        # # Calling Method
        # self.plot_grid_search(gs.cv_results_, torch.nn.lr, torch.nn.criterion, 'lr', 'Loss Function')


# Nilearn_Regression
#     def nilearn(self, X_train, X_test, y_train, y_test, shape=(256, 256, 256), loss_function='graph-net'):
#         template = load_mni152_template()
#
#         # Pre-process the Images (may want to do more of this)
#
#         new_X_train = []
#         for x in X_train:
#             z = resample_to_img(x, template)
#             z_affine = z.affine
#             c = resample_img(x, target_affine=z_affine, target_shape=shape)
#             new_X_train.append(c)
#         new_X_test = []
#         for x in X_test:
#             z = resample_to_img(x, template)
#             z_affine = z.affine
#             c = resample_img(x, target_affine=z_affine, target_shape=shape)
#             new_X_test.append(c)
#
#         print("Resamplng Complete")
#         all_images = np.append(X_train, X_test)
#         background_img = nilearn.image.mean_img(all_images)
#         print("Background image computed with which to plot coefs")
#
#         # The model
#         if self.task == "classification":
#             decoder = SpaceNetClassifier(memory="nilearn_cache", penalty=loss_function,
#                                     screening_percentile=5., memory_level=1)
#         else:
#             decoder = SpaceNetRegressor(memory="nilearn_cache", penalty=loss_function,
#                                     screening_percentile=5., memory_level=1)
#
#         # Fit the model
#         decoder.fit(new_X_train, y_train)
#
#         # Coef_img is the points to plot on brain for visualization
#         coef_img = decoder.coef_img_
#
#         # Predictions on test set
#         if self.task == "classification":
#             y_pred = np.round(decoder.predict(new_X_test).ravel())
#
#             accuracy = np.average([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))])
#             print("ACCURACY: ", accuracy)
#             suptitle = "Accuracy: " + str(accuracy)
#             score = accuracy
#         else:
#             y_pred = decoder.predict(new_X_test).ravel()
#
#             # Evaluation Metrics
#             mse = np.mean(np.abs(y_test - y_pred))
#             score = mse
#             r2 = r2_score(y_test, y_pred)
#             print('Mean square error (MSE) on the predicted Cudit Score: %.2f' % mse)
#             print('R2 Score on the predicted Cudit Score: %.2f' % r2)
#             suptitle = loss_function + " MAE: " + str(mse) + " r2: " + str(r2)
#
#         # Plot
#         plt.figure()
#         plt.suptitle(suptitle)
#         linewidth = 3
#         ax1 = plt.subplot('211')
#         ax1.plot(y_test, label="True Cudit Score", linewidth=linewidth)
#         ax1.plot(y_pred, '--', c="g", label="Predicted Cudit Score", linewidth=linewidth)
#         ax1.set_ylabel("Cudit Score")
#         plt.legend(loc="best")
#         ax2 = plt.subplot("212")
#         ax2.plot(y_test - y_pred, label="True - predicted",
#                  linewidth=linewidth)
#         ax2.set_xlabel("subject")
#         plt.legend(loc="best")
#
#         title = loss_function + "weights"
#         plot_stat_map(coef_img, background_img, title=title)
#
#         plt.show()
#
#         return score