# -*- coding: utf-8 -*-
"""
Program name: graf_results_optimized.py
-------------------------------------------------------
RHONN EEG Denoising Visualization and MSE Evaluation Script - Optimized

This script:
1. Creates result folders if they do not exist.
2. Loads test EEG signals and RHONN-denoised data (.npy version).
3. Computes Mean Squared Error (MSE) for each channel.
4. Plots and saves Ground Truth, Noisy, and Denoised signals.
5. Saves a CSV file with the MSE results.

IMPORTANT:
To compare results with test_optimized.py, you MUST select the same 
'dataset_option' and 'selected_combination' values here as in test_optimized.py.
If they do not match, the plotted signals and computed MSE will NOT correspond.

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
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- Create results folders --------------------
folder_name = 'RHONN_results'
folder_images = os.path.join(folder_name, 'images')
os.makedirs(folder_name, exist_ok=True)
os.makedirs(folder_images, exist_ok=True)

# -------------------- Parameters --------------------
# =====================================================
# WARNING:
# Select the same 'dataset_option' and 'selected_combination'
# here and in test_optimized.py.
# If they do not match, the plotted signals and MSE results
# will not correspond to the denoised data generated.
# =====================================================

dataset_option = 'original'  # Options: 'original', '30pct', '50pct', '70pct', '90pct'
combinations = ['eeg_emg', 'eeg_eog', 'eeg_eog_emg', 'eeg_heart']
selected_combination = 'eeg_emg'

num_signals_to_plot = 10  # Number of EEG channels to plot

# -------------------- Load EEG data --------------------
if dataset_option == 'original':
    EEG_all_GT = sio.loadmat('data/preprocesing_data/EEG_all_GT_norma_test.mat')['EEG_all_GT_norma_test']
    EEG_noise  = sio.loadmat('data/preprocesing_data/EEG_noisy_norma_test.mat')['EEG_noisy_norma_test']
else:
    base_path = f'data/real_noisy__data/P_{dataset_option}/{selected_combination}/FormatoMatlab'
    EEG_all_GT = sio.loadmat(os.path.join(base_path, 'EEG_all_GT_norma_test.mat'))['EEG_all_GT_norma_test']
    EEG_noise  = sio.loadmat(os.path.join(base_path, 'EEG_noisy_norma_test.mat'))['EEG_noisy_norma_test']

# -------------------- Load denoised EEG signals --------------------
Denoised_data_RHONN = np.load(os.path.join(folder_name, 'Denoising_data_RHONN.npy'))

# -------------------- Parameters for plotting --------------------
num_channels = EEG_all_GT.shape[1]
mse_list = np.zeros(num_channels)
num_signals_to_plot = min(num_signals_to_plot, num_channels)

# -------------------- Loop to compute MSE and plot signals --------------------
for i in range(num_signals_to_plot):
    gt = EEG_all_GT[:, i]
    x  = EEG_noise[:, i]
    y  = Denoised_data_RHONN[:, i]

    # Compute MSE
    mse_list[i] = np.mean((gt - y) ** 2)

    # Plot signals
    plt.figure(figsize=(18, 9))
    plt.plot(gt, 'k', linewidth=2, label='Clean Signal')
    plt.plot(x, 'b', linewidth=1.5, label='Noisy Signal')
    plt.plot(y, 'r', linewidth=1.5, label='RHONN + EKF Denoised')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'EEG Signal {i+1}')
    plt.grid(True)

    # Save figure
    plt.savefig(os.path.join(folder_images, f'plot_signal_{i+1}.png'))
    plt.close()

# -------------------- Compute MSE for all channels --------------------
for i in range(num_channels):
    gt = EEG_all_GT[:, i]
    y  = Denoised_data_RHONN[:, i]
    mse_list[i] = np.mean((gt - y) ** 2)

# -------------------- Save MSE results --------------------
channels = np.arange(1, num_channels + 1)
results_table = pd.DataFrame({'Signal': channels, 'MSE': mse_list})

# Add average MSE row
average_mse = np.mean(mse_list)
results_table = pd.concat([results_table,
                           pd.DataFrame({'Signal': [0], 'MSE': [average_mse]})],
                          ignore_index=True)

# Save CSV
results_table.to_csv(os.path.join(folder_name, 'MSE_results.csv'), index=False)

print(f"Plots saved for {num_signals_to_plot} signals and MSE results exported successfully.")
print(f"[INFO] dataset_option = {dataset_option}")
print(f"[INFO] selected_combination = {selected_combination}")
print("[REMINDER] These values must match those in test_optimized.py for valid comparison.")
