# EEG Seizure Detection and Prediction usaing Machine learning (FYP)

## Project Overview

This project focuses on detecting and predicting epileptic seizures using EEG signals and machine learning techniques.

The main goal of this system is to:
- Detect seizure-related patterns from EEG recordings
- Predict seizures before they occur (preictal stage)
- Process raw EEG signals into machine learning-ready data
- Train deep learning models capable of learning temporal EEG patterns

The project uses a full EEG processing and deep learning pipeline, starting from raw EEG recordings and ending with models that can help predict seizures.

The system includes:
- EEG preprocessing
- Signal filtering
- Window segmentation
- Preictal/interictal labeling
- Frequency feature extraction
- CNN baseline models, hYBRID cnn + Frequency model Also CNN-LSTM temporal sequence model
---

## Dataset

This project uses the CHB-MIT Scalp EEG Dataset.

The dataset contains EEG recordings from multiple patients with seizure annotations.
Dataset link:
https://physionet.org/content/chbmit/1.0.0/

Note: The dataset is not included in this repository because it is very large.

---

## Features

- EEG preprocessing pipeline using MNE
- Bandpass and notch signal filtering
- FINAL_17 EEG channel standardization
- EEG window segmentation with overlap
- Seizure and preictal labeling
- Frequency-domain feature extraction
- CNN baseline model
-  Hybrid CNN + Frequency model
- CNN-LSTM sequence prediction model
- Patient-wise train/validation/test splitting
- Balanced training batch generation
- Cached batch generation system
- Real-world unbalanced test evaluation

---

## Models

1. Baseline CNN Model
The base model used in this project is a Convolutional Neural Network (CNN).

- Input: EEG windows (channels × time samples)  
- Output: Seizure / Non-seizure
2. CNN + Frequency Hybrid Model
This model combines:

- Raw EEG signals
- Frequency-domain EEG features
  
The CNN branch learns temporal waveform patterns while the frequency branch learns spectral EEG information.

3. CNN-LSTM Sequence Model
The final model combines:
- CNN feature extraction
- LSTM temporal sequence learning
- Frequency-domain features
  This allows the model to learn both Speatial EEG patterns and Temporal seizure evolution over time
---
## How the system works
Raw EDF EEG Files

↓

Signal Filtering

↓

FINAL_17 Channel Standardization

↓

Window Segmentation

↓

Preictal / Interictal Labeling

↓

Frequency Feature Extraction

↓

Cached Batch Generation

↓

CNN / CNN + Frequency / CNN-LSTM Models

↓

Prediction and Evaluation

## Project Structure

EEG_FYP/

├── src/ # Main source code

├── notebooks/ # Experimentation and testing

├── models/ # Saved models

├── plots/ # Graphs and visualizations

├── results/ # Evaluation results

├── requirements.txt # Required Python libraries

├── README.md # Project documentation

---

## Cache system

The project includes a custom caching system designed to improve training speed and memory efficiency.

The cache pipeline includes:
- Balanced training caches
- Validation caches
- Frequency feature caches
- Real-world test caches

---

## Real-World Testing

Training batches are balanced so the models can better learn seizure-related patterns during training.

For final testing, natural unbalanced EEG data is used to make the evaluation more realistic and closer to real-world conditions.

This helps show how the model performs on unseen patient data in practical situations.

---
## How to Run the Project

### 1. Clone the repository


    git clone https://github.com/Montaser-Taher/EEG-Seizure-Prediction-FYP.git
    cd EEG-Seizure-Prediction-FYP

### 2. Create a virtual environment

    python -m venv .venv

### Windows:
    .venv\Scripts\activate
### Linux / Mac:
    source .venv/bin/activate
### 3. install dependencices
    pip install -r requirements.txt
### 4. Add dataset 
donwload the CHB-MIT dataset and place it inside:
data/raw/
### 5. Run preprocessing
### run all preprocessing pipelines for all groups 
    python src/preprocess.py
### 6. Train the model
    python src/train.py

## Important Notes

- EEG data files are not included  
- Processed `.npz` files are not included  
- Virtual environment (`.venv`) is not included  

This is done to keep the repository clean and lightweight.

---

## Author

Elmuntserbalah Taher  
B.Sc. Software Development (Hons)  
University of Malta  

---

## Notes

This project is part of a Final Year Project focused on applying machine learning to real-world EEG data.
