# Speech Disorder Detection Using Machine Learning and Deep Learning Techniques

This repository contains the implementation and results of our research on classifying pathological and non-pathological speech using machine learning and deep learning methods. The study demonstrates the potential of automated systems in enhancing speech disorder diagnostics, particularly in resource-constrained settings.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [MFCC-Based Feature Extraction](#mfcc-based-feature-extraction)
  - [Mel-Frequency Spectrograms and CNN](#mel-frequency-spectrograms-and-cnn)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

Speech disorders affect millions worldwide, significantly impacting communication and quality of life. Traditional diagnostic methods are often subjective, time-intensive, and reliant on specialized professionals. This study leverages machine learning and deep learning to automate speech disorder detection, providing a scalable, efficient, and accurate solution.

---

## Dataset

The dataset used in this research is the [Russian Speech Disorder Audio (RSDA) dataset](https://www.kaggle.com/datasets/mhantor/russian-voice-dataset). It includes:
- **2,000 Pathological Voice Samples**
- **2,000 Non-Pathological Voice Samples**
- Language: Russian
- Disorder: Hyperkinetic Dysarthria

The dataset is publicly available and was preprocessed to extract meaningful features for model training and evaluation.

---

## Methodology

### MFCC-Based Feature Extraction
1. **Audio Processing**: MFCC features were extracted to represent spectral properties of voice samples.
2. **Classical Models**: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest (RF) were applied to classify the samples.
3. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and Specificity were calculated.

### Mel-Frequency Spectrograms and CNN
1. **Spectrogram Generation**: Each audio sample was converted into a 2D Mel-frequency spectrogram.
2. **Deep Learning Model**: A Convolutional Neural Network (CNN) was designed and trained on the spectrogram images.
3. **Augmentation**: Techniques like noise injection and pitch shifting were applied to enhance robustness.
4. **Evaluation**: Accuracy and a confusion matrix were used to measure performance.

---

## Results

| Model                | Accuracy | Precision | Recall | F1-Score | Specificity |
|----------------------|----------|-----------|--------|----------|-------------|
| **KNN**             | 99.523%  | 1.00      | 0.89   | 0.94     | 98.45%      |
| **Random Forest**    | 99.523%  | 1.00      | 0.89   | 0.94     | 99.48%      |
| **SVM**              | 99.762%  | 1.00      | 0.95   | 0.97     | 100%        |
| **CNN**              | 99.87%   | -         | -      | -        | -           |

The CNN model achieved the highest accuracy by leveraging spectrogram data to identify subtle time-frequency patterns.

---

## Technologies Used
- Python
- Librosa
- Scikit-learn
- TensorFlow/Keras
- Matplotlib & Seaborn (for visualizations)

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
3. Run the preprocessing and model training scripts:
   ```bash
   python My_copy_of_Speech_Disorder_Analysis.py
   ```

## Future Work

- Expanding the dataset to include diverse languages and speech disorders.
- Exploring multimodal approaches (e.g., combining audio and visual data).
- Deploying the models on edge devices for real-time use.

---

## References

1. Y. Verma and R. S. Anand, "Gesture generation by the robotic hand for aiding speech and hard of hearing persons based on Indian sign language," *Heliyon*, 2024.
2. S. Gibbon and A. Attaheri, "Developing biomarkers for language disorders using EEG and machine learning," *Int. J. Speech Lang. Pathol.*, 2021.
3. S. Strombergsson and J. Edlund, "Evaluating speech acceptability in children with SSD," *J. Commun. Disord.*, 2021.
4. Dataset: [Russian Speech Disorder Audio (RSDA)](https://www.kaggle.com/datasets/mhantor/russian-voice-dataset).
5. F. Godino-Llorente et al., "SVM classification of pathological voices for diagnosis," *Speech Commun.*, 2005.

---

