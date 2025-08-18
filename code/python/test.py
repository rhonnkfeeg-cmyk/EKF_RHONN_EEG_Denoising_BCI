# -*- coding: utf-8 -*-
"""
Program name: test_optimized.py
-------------------------------------------------------
RHONN EEG Denoising Script - Python version

This script:
1. Loads normalized test EEG signals (clean and noisy).
2. Applies RHONN (Recurrent High-Order Neural Network) denoising using pre-trained weights.
3. Stores the denoised EEG signals in both MATLAB (.mat) and NumPy (.npy) formats.

Based on:
Denoising of EEG Signals in Brain–Computer Interfaces Using 
Kalman-Filtered High-Order Recurrent Neural Networks [Reference: Add article here]

Minimum requirements:
- Python version: 3.7
- numpy version: 1.21.6
- scipy version: 1.7.3
- matplotlib version: 3.5.3 (optional, for plotting)
- pandas version: 1.3.5 (optional, for tabular storage)

Authors: Martínez Madrid, Enrique
         Medrano Hermosillo, Jesús Alfonso
         Ramírez Quintana, Juan Alberto
         Rodríguez Mata, Abraham Efraim
         González Huitrón, Victor Alejandro
         Urbina Leos, Iván Ramón
"""

import os
import numpy as np
import scipy.io as sio

# -------------------- Create results folder --------------------
folder_name = 'RHONN_results'
os.makedirs(folder_name, exist_ok=True)

# -------------------- Parameters --------------------
# =====================================================
# WARNING:
# Select the same 'dataset_option' and 'selected_combination'
# here and in graf_results_optimized.py.
# If they do not match in both scripts, the signals and MSE values
# will not correspond, and you will not be able to compare results.
# =====================================================

# Options: 'original', '30pct', '50pct', '70pct', '90pct'
dataset_option = 'original'  # Change to '30pct', '50pct', etc. for noisy data

# If dataset_option != 'original', select the combination
combinations = ['eeg_emg', 'eeg_eog', 'eeg_eog_emg', 'eeg_heart']
selected_combination = 'eeg_emg'

# -------------------- Load EEG data --------------------
if dataset_option == 'original':
    EEG_all_GT = sio.loadmat('data/preprocesing_data/EEG_all_GT_norma_test.mat')['EEG_all_GT_norma_test']
    EEG_noise  = sio.loadmat('data/preprocesing_data/EEG_noisy_norma_test.mat')['EEG_noisy_norma_test']
else:
    base_path = f'data/real_noisy__data/P_{dataset_option}/{selected_combination}/FormatoMatlab'
    EEG_all_GT = sio.loadmat(os.path.join(base_path, 'EEG_all_GT_norma_test.mat'))['EEG_all_GT_norma_test']
    EEG_noise  = sio.loadmat(os.path.join(base_path, 'EEG_noisy_norma_test.mat'))['EEG_noisy_norma_test']

# -------------------- Denoising --------------------
num_columns = EEG_all_GT.shape[1]
Deniosing_data_RHONN = np.zeros_like(EEG_all_GT)

w = np.load('w.npy')  # Load weights once

for i in range(num_columns):
    x = EEG_noise[:, i]
    N = len(x)
    y = np.zeros(N)
    y_prev1 = y_prev2 = 0.0

    for k in range(N):
        xi1 = np.tanh(y_prev1)
        xi2 = np.tanh(y_prev2)
        xi3 = x[k]
        y[k] = (w[0]*xi1 + w[1]*xi1*xi2 + w[2]*xi1*xi3 +
                w[3]*xi2*xi3 + w[4]*xi3)
        y_prev2 = y_prev1
        y_prev1 = y[k]

    Deniosing_data_RHONN[:, i] = y
    print(f"Channel {i+1} processed.")

# -------------------- Save results --------------------
sio.savemat(os.path.join(folder_name, 'Denoising_data_RHONN.mat'),
            {'Denoising_data_RHONN': Deniosing_data_RHONN})
np.save(os.path.join(folder_name, 'Denoising_data_RHONN.npy'), Deniosing_data_RHONN)

print("Denoising completed successfully.")
print(f"[INFO] dataset_option = {dataset_option}")
print(f"[INFO] selected_combination = {selected_combination}")
print("[REMINDER] Use these same values in graf_results_optimized.py to compare results.")

