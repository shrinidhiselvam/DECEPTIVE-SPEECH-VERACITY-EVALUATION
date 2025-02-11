# DECEPTIVE-SPEECH-VERACITY-EVALUATION

# Speech Emotion Recognition using RAVDESS Dataset

This project focuses on building a Speech Emotion Recognition (SER) system using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The goal is to classify audio clips into different emotional categories such as angry, disgust, fear, happy, neutral, sad, and surprise.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features Extraction](#features-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Speech Emotion Recognition (SER) is a field of study that focuses on identifying human emotions from speech signals. This project uses the RAVDESS dataset, which contains audio recordings of actors expressing different emotions. The project involves extracting features from the audio files, training a machine learning model, and evaluating its performance.

## Dataset
The RAVDESS dataset contains 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements in a neutral North American accent. The dataset includes speech and song, but this project focuses on the speech portion. The emotions included are:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset is available on [Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio).

## Features Extraction
The following features are extracted from the audio files:
 **Mel-Frequency Cepstral Coefficients (MFCCs)**: These are coefficients that collectively make up an MFC, which is a representation of the short-term power spectrum of a sound.
**Mean of MFCCs**: The mean of the MFCCs across time is used as a feature vector.

## Model Training
The extracted features are used to train a machine learning model. The dataset is split into training and testing sets, and a classifier is trained on the training set. The model used in this project is a simple classifier, but it can be replaced with more advanced models like Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs).

## Evaluation
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the performance of the classifier.



## Results
The model's performance is summarized in the classification report and confusion matrix. The results show the accuracy and other metrics for each emotion class.

Classification Report:
              precision    recall  f1-score   support

       angry       0.71      0.31      0.43        16
     disgust       0.47      0.71      0.56        31
        fear       0.62      0.48      0.54        48
       happy       0.53      0.50      0.51        36
     neutral       0.60      0.46      0.52        39
         sad       0.55      0.63      0.59        35
    surprise       0.59      0.68      0.63        40

    accuracy                           0.57       288
   macro avg       0.58      0.55      0.55       288
weighted avg       0.58      0.57      0.56       288


## Future Work
- **Improve Feature Extraction**: Explore additional features such as chroma, spectral contrast, and tonnetz.
- **Advanced Models**: Implement deep learning models like CNNs or RNNs for better performance.
- **Real-time Emotion Recognition**: Develop a real-time system that can classify emotions from live audio input.

