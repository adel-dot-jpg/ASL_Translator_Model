# ASL Hand Sign Recognition

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Project Overview

This model recognizes 24 letters of the American Sign Language alphabet (excluding J and Z, which require motion) and includes an additional 25th "unknown" class to reject non-ASL hand signs and gestures. This model utilizes a convolutional neural network built with PyTorch to classify American Sign Language (ASL) hand signs from images, with an additional "unknown" class for functionality with live services. Adapted from a basic sign_mnist CNN for use with static image input for use with another of my projects, a live ASL webcam translator.

## Tech Stack

- **Python 3.13.5**
- **PyTorch** - Deep learning framework
- **torchvision** - Image transformations and data augmentation
- **NumPy** - Data processing
- **Pandas** - CSV data handling
- **Pillow (PIL)** - Image processing

## Required Datasets

### ASL Sign Language Dataset

- [**sign_mnist**](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data)
- [**Hands and palm images dataset**](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset)
- [**Hand Gesture Recognition Database**](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

data file structure is as follows:

```
Data:
	ASL_Data:
		sign_mnist_train.csv
		sign_mnist_valid.csv
	Non_ASL_Data:
		unknown_gestures:
			train:
				(folders 00 through 07 from the .zip file)
			valid:
				(folder 08 and 09 from the .zip file)
		unknown_hands:
			train:
				(images 2260 through 11744 from the inner folder from the .zip file)
			valid:
				(images 2 through 2259 from the outer folder from the .zip file)
```

### Key Features

- **Robust Multi-source data handling**: Combines CSV-based training data with image-based unknown class data
- **Data augmentation**: A selection of appropriate randomized transformations to the corpus to improve generalization
- **Custom CNN architecture**: Three convolutional blocks with batch normalization and dropout for robust feature extraction
- **Unknown class detection**: Identifies and classifies hand gestures that are not part of the ASL alphabet

### Model Architecture

- **Input**: 28×28 grayscale images
- **Conv Block 1**: 1 → 25 channels (14×14 output)
- **Conv Block 2**: 25 → 50 channels with 20% dropout (7×7 output)
- **Conv Block 3**: 50 → 75 channels (3×3 output)
- **Fully Connected**: 675 → 512
- **Fully Connected**: 512 → 25 output classes (24 stationary ASL letters + unknown/other class)
