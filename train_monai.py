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
from nilearn.decoding import SpaceNetRegressor
from nilearn.image import smooth_img, resample_img, load_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from nilearn.plotting import show
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import resnet as rs
from AlexNet import AlexNet3D
from File_Structure import File_Structure
from sklearn.model_selection import KFold


class train_monai:
    def __init__(self, epochs=2, task='control', model_type= "nilearn", pytorch_version = 0, loss_functions = [], optimizers = [], learning_rates = [], cv=3):
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

        self.file_structure = dict()
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        # Based on Parameters
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

    def cv_grid_search(self, X, y):

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
                        self.monai_train()
                        score = self.monai_eval()

                        scores[counter][key] = score
                        print("SCORE: ", score)

            counter += 1

        stringified_dict = json.loads(json.dumps(scores), parse_int=str)
        print(stringified_dict)
        outF = open(self.saved_model_dict.replace('.pth','.txt'), "w")
        outF.write(str(stringified_dict))
        outF.close()


    def evaluate(self, subset, fraction, kernel='linear', penalty='graph-net'):

        score = 0
        # shape = the shape of MRI. Since BL and FU have different shapes, must standardize when using nilearn
        if subset == "all":
            shape = (256, 256, 256)
        elif subset == "FU":
            shape = (256, 256, 170)
        else:
            shape = (256, 182, 256)

        # The train/eval data specialized for either regression or classification
        images, labels = self.File_Structure.model_input(subset, fraction)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, shuffle=False)
        self.setup(X_train, X_test, y_train, y_test, 1e-5, torch.optim.SGD, torch.nn.CrossEntropyLoss())
        X = np.append(X_train, X_test)
        y = np.append(y_train, y_test)

        if self.cv > 0:
            self.cv_grid_search(X,y)

        # Run model
        if self.model_type == "nilearn":

            if self.task == "regression" or self.task == "classification":
                score = self.nilearn(X_train, X_test, y_train, y_test, shape, penalty)

            elif self.task == "segmentation":
                score = self.nilearn_SVM(X, y, shape, kernel)

        elif self.model_type == "monai":

            if self.task == "classification":
                self.monai_train()
                score = self.monai_eval()

            else:
                print("not yet made, need to adapt")

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
            self.model = rs.resnet200()
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
    def monai_train(self):

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
                        torch.save(self.model.state_dict(), self.saved_model_dict)
                    print(
                        "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                            epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()

    # Evaluates best model that was saved
    # Only for classification right now
    def monai_eval(self):
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
            print("evaluation metric:", metric)
            saver.finalize()


            return [metric, binary_]

    # Segmentation
    def nilearn_SVM(self,  X, y, shape, kernel='linear'):

        # Pre-processing
        resampled_X = []
        for x in X:
            resampled_X.append(resample_img(x, target_affine=np.eye(4), target_shape=shape))
        masker = NiftiMasker(smoothing_fwhm=4,
                                standardize=True, memory="nilearn_cache", memory_level=1)
        X = masker.fit_transform(resampled_X)

        # Model
        svc = SVC(kernel=kernel)
        #pca = PCA(svd_solver='full', n_components=0.95) tru tjos with pca.fit_transform # TO TRY LATER
        feature_selection = SelectPercentile(f_classif, percentile=5)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
        anova_svc.fit(X, y)
        y_pred = anova_svc.predict(X)
        print(y_pred)

        # Compute the prediction accuracy for the different folds (i.e. session)
        cv_scores = cross_val_score(anova_svc, X, y, cv = 8)
        print(cv_scores)

        # Return the corresponding mean prediction accuracy
        classification_accuracy = cv_scores.mean()
        print(classification_accuracy)
        print("Classification accuracy: %.4f / Chance level: %f" %
              (classification_accuracy, 1. / 2.))

        coef = svc.coef_

        # reverse feature selection
        coef = feature_selection.inverse_transform(coef)

        # reverse masking
        weight_img = masker.inverse_transform(coef)

        # Use the mean image as a background to avoid relying on anatomical data
        mean_img = nilearn.image.mean_img(resampled_X)

        # Create the figure
        plot_stat_map(weight_img, mean_img, title='SVM weights')

        show()

        return classification_accuracy

    # Nilearn_Regression
    def nilearn(self, X_train, X_test, y_train, y_test, shape=(256, 256, 256), loss_function='graph-net'):

        # Pre-process the Images (may want to do more of this)

        new_X_train = []
        for x in X_train:
            a = load_img(x)
            b = smooth_img(a, fwhm=5)
            c = resample_img(b, target_affine=np.eye(4), target_shape=shape)
            new_X_train.append(c)
        new_X_test = []
        for x in X_test:
            a = load_img(x)
            b = smooth_img(a, fwhm=5)
            c = resample_img(b, target_affine=np.eye(4), target_shape=shape)
            new_X_test.append(c)

        print("Resamplng Complete")

        background_img = new_X_train[0]
        print("Background image computed with which to plot coefs")

        # The model
        decoder = SpaceNetRegressor(memory="nilearn_cache", penalty=loss_function,
                                    screening_percentile=5., memory_level=1)

        # Fit the model
        decoder.fit(new_X_train, y_train)

        # Coef_img is the points to plot on brain for visualization
        coef_img = decoder.coef_img_

        # Predictions on test set
        if self.task == "classification":
            y_pred = np.round(decoder.predict(new_X_test).ravel())

            accuracy = np.average([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))])
            print("ACCURACY: ", accuracy)
            suptitle = "Accuracy: " + str(accuracy)
            score = accuracy
        else:
            y_pred = decoder.predict(new_X_test).ravel()

            # Evaluation Metrics
            mse = np.mean(np.abs(y_test - y_pred))
            score = mse
            r2 = r2_score(y_test, y_pred)
            print('Mean square error (MSE) on the predicted Cudit Score: %.2f' % mse)
            print('R2 Score on the predicted Cudit Score: %.2f' % r2)
            suptitle = loss_function + " MAE: " + str(mse) + " r2: " + str(r2)

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

        title = loss_function + "weights"
        plot_stat_map(coef_img, background_img, title=title,
                      display_mode="y", dim=-.5)
        plot_stat_map(coef_img, background_img, title=title,
                       display_mode="x", dim=-.5)
        plot_stat_map(coef_img, background_img, title=title,
                       display_mode="z", dim=-.5)
        #plot_stat_map(coef_img, background_img, title="graph-net weightsy", display_mode="y", cut_coords=1, dim=-.5)
        #plot_stat_map(coef_img, background_img, title="graph-net weightsz",
                      #display_mode="z", cut_coords=1, dim=-.5)
        #plot_stat_map(coef_img, background_img, title=title, dim=-.5)
        plt.show()

        # Trying to convert to MRI coordinates but doesn't work

        # niimg = nilearn.datasets.load_mni152_template()
        # converted = nilearn.image.coord_transform(19, 77, 15, background_img)
        # print(converted)

        return score

    def visualize(self, c=None):
        if c != None:
            #plot_stat_map(c)
            #show()
            plotting.plot_glass_brain(c)
            show()
        else:
            plotting.plot_glass_brain('sub-320/ses-BL/anat/sub-320_ses-BL_T1w.nii.gz')
            show()
        # # Load image
        # bg_img = nibabel.load(('sub-320/ses-BL/anat/sub-320_ses-BL_T1w.nii.gz'))
        # bg = bg_img.get_data()
        # # Keep values over 4000 as activation map
        # act = bg.copy()
        # act[act < 6000] = 0.
        #
        # # Display the background
        # plt.imshow(bg[..., 100].T, origin='lower', interpolation='nearest', cmap='gray')
        # # Mask background values of activation map
        # masked_act = np.ma.masked_equal(act, 0.)
        # #plt.imshow(masked_act[..., 10].T, origin='lower', interpolation='nearest', cmap='hot')
        # # Cosmetics: disable axis
        # plt.axis('off')
        # plt.show()
        # # Save the activation map
        # #nibabel.save(nibabel.Nifti1Image(act, bg_img.get_affine()), 'activation.nii.gz')
        # Auxiliary Function

    def flatten_list(self, x):
        y = []
        for i in x:
            for j in i:
                y.append(j)
        return y

    def binary_classification(self, y_true, y_pred):
        print("Classification Results: ")
        print(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        output = "TN:" + str(tn) + "FP:" + str(fp) + "FN:" + str(fn) + "TP:" + str(tp)
        print(output)
        return output

        # y_pred_class = y_pred_pos > threshold
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        # false_positive_rate = fp / (fp + tn)

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
