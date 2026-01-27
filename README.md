# MNIST Bootstrap Project 🎓

A Python bootstrap project for learning artificial intelligence with the MNIST dataset. This project explores two different approaches to handwritten digit recognition: Convolutional Neural Networks (CNN) and K-Nearest Neighbors (KNN).

## Overview

This educational project is designed to help students understand machine learning fundamentals by working with the famous MNIST dataset of handwritten digits (0-9). The project implements and compares two distinct machine learning approaches:

1. **Convolutional Neural Network (CNN)**: A deep learning approach using TensorFlow/Keras that learns spatial hierarchies of features through convolutional layers.
2. **K-Nearest Neighbors (KNN)**: A traditional machine learning algorithm that classifies digits based on similarity to training examples.

## Features

- Clean, modular code structure with separation of concerns
- Automatic model training with early stopping to prevent overfitting
- Model persistence for reusing trained models
- Comprehensive evaluation with confusion matrices
- Support for both training and prediction pipelines
- Visualization utilities for data exploration and results analysis

## Requirements

- Python 3.x
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Installation

1. Clone this repository:
```bash
git clone https://github.com/N3ik0/python_bootstrap.git
cd python_bootstrap
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the MNIST dataset files in the `data/` directory:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`

## Project Structure

```
python_bootstrap/
├── src/
│   ├── models/
│   │   ├── cnn_model.py       # CNN architecture definition
│   │   └── __init__.py
│   ├── config.py              # Project configuration and paths
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── train.py               # Training pipeline
│   ├── predict.py             # Prediction pipeline
│   ├── utils.py               # Utility functions and visualizations
│   └── __init__.py
├── legacy_backup/
│   └── models/
│       ├── CNN/               # Legacy CNN implementation
│       └── KNN/               # Legacy KNN implementation
├── data/                      # MNIST dataset files
├── saved_models/              # Trained model files
├── main_train.py              # Entry point for training
├── main_predict.py            # Entry point for prediction
└── requirements.txt           # Project dependencies
```

## Usage

### Training the CNN Model

To train a new CNN model or continue training an existing one:

```bash
python main_train.py
```

This will:
- Load and preprocess the MNIST dataset
- Create or load an existing CNN model
- Train the model with early stopping (patience=3)
- Save the trained model to `saved_models/mnist_cnn_model.keras`
- Display training results and confusion matrix

### Making Predictions

To use a trained model for predictions:

```bash
python main_predict.py
```

This will:
- Load the trained CNN model
- Evaluate the model on the test set
- Display accuracy metrics and confusion matrix
- Show examples of prediction errors (if any)

### Working with Individual Models

#### CNN Model

The CNN model uses a sequential architecture with:
- 2 convolutional layers (32 and 64 filters)
- MaxPooling layers for dimension reduction
- Dense layers for classification
- Softmax activation for probability distribution over 10 classes

Training features:
- Adam optimizer
- Sparse categorical cross-entropy loss
- Early stopping with validation monitoring
- Automatic model checkpointing

#### KNN Model (Legacy)

The KNN implementation is available in `legacy_backup/models/KNN/`:
- Uses scikit-learn's KNeighborsClassifier
- Works with flattened image vectors (784 features)
- Configurable number of neighbors (default: k=3)

## Model Performance

The CNN model typically achieves:
- Test accuracy: 98-99%
- Fast inference time
- Robust performance on various digit styles

The KNN model typically achieves:
- Test accuracy: 96-97%
- Longer inference time for large datasets
- Good baseline for comparison

## Data Preprocessing

The project includes two preprocessing strategies:

1. **For CNN**: Images are reshaped to (28, 28, 1) to preserve spatial structure and normalized to [0, 1].
2. **For KNN**: Images are flattened to 784-dimensional vectors and normalized to [0, 1].

## Visualization Tools

The project includes utilities for:
- Displaying sample digits from the dataset
- Showing digit distribution across training and test sets
- Generating confusion matrices to analyze model performance
- Visualizing prediction errors

## Learning Objectives

This project helps students learn:
- How to load and preprocess image data
- The difference between traditional ML (KNN) and deep learning (CNN) approaches
- How to build and train neural networks with TensorFlow/Keras
- Model evaluation techniques and metrics
- The importance of data normalization and preprocessing
- How to save and load trained models
- Best practices for code organization in ML projects

## Configuration

Project settings can be modified in `src/config.py`:
- Data directory paths
- Model save locations
- Dataset file paths

## Contributing

This is an educational project. Feel free to fork and experiment with:
- Different CNN architectures
- Hyperparameter tuning
- Additional ML algorithms
- Data augmentation techniques
- Advanced visualization methods

## License

This project is intended for educational purposes.

## Acknowledgments

- MNIST dataset provided by Yann LeCun and collaborators
- Built with TensorFlow, Keras, and scikit-learn
