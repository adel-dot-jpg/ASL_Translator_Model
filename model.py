import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
NUM_ASL_CLASSES = 24   # sign_mnist excludes J and Z
UNKNOWN_CLASS = 24     # last unkown class
N_CLASSES = 25		   # total classes

train_df = pd.read_csv("4. Data Augmentation/Data/ASL_Data/sign_mnist_train.csv")
valid_df = pd.read_csv("4. Data Augmentation/Data/ASL_Data/sign_mnist_valid.csv")

class MyDataset(Dataset):
    def __init__(self, base_df):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.values / 255  # Normalize values from 0 to 1
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)

class UnknownHandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg"): # remove jpeg? keep just to be safe
            self.image_paths.extend(self.root_dir.rglob(ext))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)

        label = UNKNOWN_CLASS
        return img, label

n = 32
train_data = MyDataset(train_df)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)

valid_data = MyDataset(valid_df)
valid_loader = DataLoader(valid_data, batch_size=n)
valid_N = len(valid_loader.dataset)

# custom module
#__init__: defines any properties we want our module to have, including our neural network layers.
# We will effectively be using a model within a model.
#forward: defines how we want the module to process any incoming data from the previous layer it is connected to.
# Since we are using a Sequential model, we can pass the input data into it like we are making a prediction.
class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

# same model utilizing custom module this time
flattened_img_size = 75 * 3 * 3

# Input 1 x 28 x 28
base_model = nn.Sequential(
    MyConvBlock(IMG_CHS, 25, 0), # 25 x 14 x 14
    MyConvBlock(25, 50, 0.2), # 50 x 7 x 7
    MyConvBlock(50, 75, 0),  # 75 x 3 x 3
    # Flatten to Dense Layers
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512, N_CLASSES)
)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(base_model.parameters())

# compile and send to GPU (requies c compiler available)
#model = torch.compile(base_model.to(device))
# send to GPU (no c compiler available)
model = base_model.to(device)
# print model now shows not only the custom module layer but also its contained module layers
#print(model)

# now using torchvision to augment our data

# sample image for testing using different transformations
# row_0 = train_df.head(1)
# y_0 = row_0.pop('label')
# x_0 = row_0.values / 255
# x_0 = x_0.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
# x_0 = torch.tensor(x_0)
# x_0.shape

# image = F.to_pil_image(x_0)
# plt.imshow(image, cmap='gray')

# trans = transforms.Compose([
#     transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.7, 1), ratio=(1, 1)),
# ])

# new_x_0 = trans(x_0)
# image = F.to_pil_image(new_x_0)
# plt.imshow(image, cmap='gray')

# trans = transforms.Compose([
#     transforms.RandomHorizontalFlip()
# ])

# new_x_0 = trans(x_0)
# image = F.to_pil_image(new_x_0)
# plt.imshow(image, cmap='gray')

# trans = transforms.Compose([
#     transforms.RandomRotation(10)
# ])

# new_x_0 = trans(x_0)
# image = F.to_pil_image(new_x_0)
# plt.imshow(image, cmap='gray')


# brightness = .2  # Change to be from 0 to 1
# contrast = .5  # Change to be from 0 to 1

# trans = transforms.Compose([
#     transforms.ColorJitter(brightness=brightness, contrast=contrast)
# ])

random_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.2, contrast=.5)
])

# new_x_0 = random_transforms(x_0) # test transforms
# image = F.to_pil_image(new_x_0)
# plt.imshow(image, cmap='gray')

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(random_transforms(x))  # Updated
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += utils.get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += utils.get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


# training
epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

# the validation accuracy is higher, and more consistent.
# This means that our model is no longer overfitting in the way it was;
# it generalizes better, making better predictions on new data.

#The training accuracy may be lower, and that's ok.
# Compared to before, the model is being exposed to a much larger variety of data.

# saving the model to disk
#torch.save(base_model, 'model.pth') # saves whole model, not recommended

torch.save(model.state_dict(), "model_weights.pth") # saves weights only, safer for web