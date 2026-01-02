import torch.nn as nn
import pandas as pd
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from pathlib import Path
from PIL import Image

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
NUM_ASL_CLASSES = 24   # sign_mnist excludes J and Z
UNKNOWN_CLASS = 24     # final class (unknown)
N_CLASSES = 25		   # total classes

train_df = pd.read_csv("Data/ASL_Data/sign_mnist_train.csv")
valid_df = pd.read_csv("Data/ASL_Data/sign_mnist_valid.csv")

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
		for ext in ("*.jpg", "*.png", "*.jpeg"): # keep jpeg just to be safe
			self.image_paths.extend(self.root_dir.rglob(ext))
		
	def __getitem__(self, idx):
		img = Image.open(self.image_paths[idx]).convert("L")  # grayscale
		img = img.resize((IMG_WIDTH, IMG_HEIGHT)) # to match sign_mnist image size
		img_array = np.array(img)
		x_df = img_array / 255  # Normalize values from 0 to 1
		x_df = x_df.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
		x = torch.tensor(x_df).float().to(device)
		y = torch.tensor(UNKNOWN_CLASS).to(device) # 24
		return x, y

	def __len__(self):
		return len(self.image_paths)

n = 32
ASL_dataset = MyDataset(train_df) # valid asl data

hands_dataset = UnknownHandDataset( # invalid hands data
	root_dir="Data/Non_ASL_Data/unknown_hands/Hands/train"
)

gestures_dataset = UnknownHandDataset( # invalid gestures data
	root_dir="Data/Non_ASL_Data/unknown_gestures/train"
)

unknown_dataset = ConcatDataset([ # all invalid data
	hands_dataset,
	gestures_dataset
])

train_data = ConcatDataset([ # all train data
	ASL_dataset,
	unknown_dataset
])

train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)


valid_ASL_dataset = MyDataset(valid_df) # valid asl validation data

valid_hands_dataset = UnknownHandDataset( # invalid hands validation data
	root_dir="Data/Non_ASL_Data/unknown_hands/Hands/valid"
)

valid_gestures_dataset = UnknownHandDataset( # invalid gestures validation data
	root_dir="Data/Non_ASL_Data/unknown_gestures/valid"
)

valid_unknown_dataset = ConcatDataset([ # all invalid validation data
	valid_hands_dataset,
	valid_gestures_dataset
])

valid_data = ConcatDataset([ # all validation data
	valid_ASL_dataset,
	valid_unknown_dataset
])

valid_loader = DataLoader(valid_data, batch_size=n)
valid_N = len(valid_loader.dataset)

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

model = base_model.to(device)


# now using torchvision to augment our data
random_transforms = transforms.Compose([
	transforms.RandomRotation(5),
	transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(brightness=.2, contrast=.5)
])

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

torch.save(model.state_dict(), "model_weights.pth") # saves weights only, safer for web