# EKF_RHONN_EEG_Denoising_BCI

## Introduction

This repository presents a program that demonstrates a **Recurrent High-Order Neural Network (RHONN)** model, proposed for **noise reduction in electroencephalographic (EEG) signals**. Designed for **Brain-Computer Interface (BCI) applications**, the method enhances signal quality by effectively identifying and suppressing artifacts and noise in dynamic environments.

The RHONN is trained using the **Extended Kalman Filter (EKF)**, which provides an advantage over other techniques, as the EKF minimizes Gaussian noise through optimized gain updates, thereby improving the performance of the RHONN architecture.

The program is developed in **MATLAB** and **Python**, and the construction of the model and RHONN architecture is fully explained and detailed in the article:  

**“Denoising of EEG Signals in Brain–Computer Interfaces Using Extended Kalman-Filtered Recurrent High-Order Neural Networks”**  
**Authors:** Martínez Madrid, Enrique; Medrano Hermosillo, Jesús Alfonso; Ramírez Quintana, Juan Alberto; Rodríguez Mata, Abraham Efraim; González Huitrón, Victor Alejandro; Urbina Leos, Iván Ramón

---

## Features

- **Recurrent High-Order Neural Network (RHONN):** Captures nonlinearities in EEG signals.  
- **Noise Reduction:** Effectively suppresses artifacts and Gaussian noise.  
- **Extended Kalman Filter (EKF) Training:** Optimized weight updates improve performance.  
- **Cross-Platform:** Works in both MATLAB and Python.  

---

## Usage Instructions

### 1. Data and Code Preparation

1. **Download** the `data` folder from the following Google Drive link:  
   [https://drive.google.com/drive/folders/1-xvTAZC7ukYPA3LK4PQhtnoOiz9Ka4TN?usp=sharing](https://drive.google.com/drive/folders/1-xvTAZC7ukYPA3LK4PQhtnoOiz9Ka4TN?usp=sharing)

2. **Download** the `code` folder from this GitHub repository.

3. **Place** the `data` folder inside:
   - `code/python`
   - `code/matlab`

### 2. Running in MATLAB

- Compatible with any version (tested on MATLAB 2020b).

**Execution order:**

1. Run `training.m` → generates the training file.  
2. Run `test.m` → performs inferences according to the desired test (see instructions inside the script).  
3. Run `results_graphs.m` → displays charts of the results (see instructions inside the script).  

### 3. Running in Python

**Requirements:**

- Python 3.7  
- numpy 1.21.6  
- scipy 1.7.3  
- matplotlib 3.5.3 *(optional, for plotting)*  
- pandas 1.3.5 *(optional, for tabular storage)*

**Execution order:**

1. Run `training.py` → generates the training file.  
2. Run `test.py` → performs inferences according to the desired test (see instructions inside the script).  
3. Run `regraf_result.py` → displays charts of the results (see instructions inside the script).  

---

## Citation

If you use this program in your research, please cite:
Martínez Madrid, E.; Medrano Hermosillo, J. A.; Ramírez Quintana, J. A.; Rodríguez Mata, A. E.; González Huitrón, V. A.; Urbina Leos, I. R. (2025). Denoising of EEG Signals in Brain–Computer Interfaces Using Extended Kalman-Filtered Recurrent High-Order Neural Networks.
