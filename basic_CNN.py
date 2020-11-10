num_classes = 10
from train_monai import *
from File_Structure import File_Structure
from nifty_file import nifty_file
import torch.nn as nn
from torch.optim import *
import torch
from torch.autograd import Variable
# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 2)
        self.conv_layer2 = self._conv_layer_set(2, 4)
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


subset = "FU"
fraction = 0.1
batch_size = 100
File_Structure = File_Structure("control")
File_Structure.organize_directory()
images, labels = File_Structure.model_input(subset, fraction)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.4, shuffle=False)

train_files = [{"img": img, "label": label} for img, label in zip(X_train, y_train)]
val_files = [{"img": img, "label": label} for img, label in zip(X_test, y_test)]

# Define transforms for image
train_transforms = Compose(
    [
        LoadNiftid(keys=["img"]),
        AddChanneld(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 96, 96)),
        RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        ToTensord(keys=["img"]),
    ]
)
val_transforms = Compose(
    [
        LoadNiftid(keys=["img"]),
        AddChanneld(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["img"]),
    ]
)
# Define dataset, data loader
check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
check_data = monai.utils.misc.first(check_loader)
print(check_data["img"].shape, check_data["label"])

# create a training data loader
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4,
                               pin_memory=torch.cuda.is_available())

# create a validation data loader
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition of hyperparameters
n_iters = 4500
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)

# Create CNN
model = CNNModel()
# model.cuda()
print(model)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):

    for batch_data in train_loader:
        images, labels = batch_data["img"].to(device), batch_data["label"].to(device)

        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        print("Happening?")
        outputs = model(images)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()

        count += 1
        if count % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                # Forward propagation
                outputs = model(val_images)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(val_labels)
                correct += (predicted == val_labels).sum()

            accuracy = 100 * correct / float(total)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
