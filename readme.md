# CEMR-LM: A domain adaptive pre-training language model for sentence classification of Chinese electronic medical record

CEMR-LM is a classification model for Chinese EMR data. It leverages a clinical domain adaptive pre-trained language Model architecture along with convolutional layers and attention mechanisms to achieve accurate classification results. This repository provides the implementation of the model along with training, testing, and evaluation scripts.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Testing](#testing)
- [Results](#results)

## Introduction

CEMR-LM is designed for classifying Chinese EMR data into multiple categories. It incorporates a clinical domain adaptive pre-trained model for text encoding, followed by convolutional layers for feature extraction and an attention mechanism to capture important information. The model's performance is evaluated using various metrics.

## Requirements

To run this code, you need the following dependencies:

- Python 3.7+
- PyTorch 1.6+
- `pytorch_pretrained_bert`
- `scikit-learn`
- `tqdm`

You can install the required packages using the following command:

pip install torch torchvision pytorch_pretrained_bert scikit-learn tqdm

## Usage

### Configuration

Before using the model, you need to configure the parameters. Modify the `Config` class in `CEMR-LM.py` to set the necessary parameters for your dataset and training preferences.

### Training

To train the model, execute the following command:

python train.py --model CEMR-LM

This will start the training process using the specified model (in this case, `CEMR-LM`). The training progress will be logged, and the best model checkpoint will be saved for later use.

This will load the best model checkpoint and evaluate it on the test dataset. The test accuracy, loss, precision, recall, and F1-score will be displayed.

## Results

The model's performance results are provided in the output logs and the test accuracy, loss, and classification report.


