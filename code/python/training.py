"""
Program name: training.py
----------------------------------------------------
This script implements a High-Order Recurrent Neural Network (RHONN)
with an Extended Kalman Filter (EKF) for training on EEG signals
with additive Gaussian noise.

Based on:
Denoising of EEG Signals in Brain–Computer Interfaces Using 
Kalman-Filtered High-Order Recurrent Neural Networks [Reference: Add article here]

Minimum requirements:
- Python version: 3.7
- numpy version: 1.21.6
- scipy version: 1.7.3
- matplotlib version: 3.5.3 (used for visualization if needed)
- pandas version: 1.3.5 (used for tabular data storage if needed)

Main steps:
1. Load clean and noisy EEG signals.
2. Initialize RHONN weights and EKF parameters.
3. Loop through each EEG channel:
   a. Predict the signal using RHONN.
   b. Compute prediction error.
   c. Update weights using EKF.
   d. Accumulate mean squared error (MSE).
4. Save final trained weights.

    Authors: Martínez Madrid, Enrique 
             Medrano Hermosillo, Jesús Alfonso
             Ramírez Quintana, Juan Alberto
             Rodríguez Mata, Abraham Efraim
             González Huitrón, Victor Alejando
             Urbina Leos, Iván Ramón
"""

import numpy as np
import scipy.io as sio

# -------------------- Load training data --------------------
# Load normalized ground truth and noisy EEG signals from .mat files
EEG_all_GT_norma_train = sio.loadmat(
    'data/preprocesing_data/EEG_all_GT_norma_train.mat'
)['EEG_all_GT_norma_train']

EEG_noisy_norma_train  = sio.loadmat(
    'data/preprocesing_data/EEG_noisy_norma_train.mat'
)['EEG_noisy_norma_train']

# Assign to working variables
EEG_all_GT = EEG_all_GT_norma_train  # Clean EEG signals
EEG_noise  = EEG_noisy_norma_train   # Noisy EEG signals

num_columnas = EEG_all_GT.shape[1]   # Number of EEG channels

# -------------------- Initialize RHONN and EKF parameters --------------------
n1 = 5                                # Number of weights in the network
P = 10 * np.eye(n1)                   # Initial covariance matrix for EKF
R = 100                                # Measurement noise variance
Q = 1e-6                               # Process noise for EKF
w = np.random.rand(n1)                # Random initial weights
msetotal = 0                           # Accumulated MSE over all channels

# -------------------- Loop through EEG channels --------------------
for i in range(num_columnas):
    gt = EEG_all_GT[:, i]  # Ground truth signal for current channel
    x  = EEG_noise[:, i]   # Noisy signal for current channel
    N  = len(x)            # Length of the signal

    # Initialize network output and previous outputs
    y = np.zeros(N)        # RHONN output
    y_prev1 = 0.0          # Previous output y(k-1)
    y_prev2 = 0.0          # Previous output y(k-2)

    # --------- Time-step loop for each sample in the channel ---------
    for k in range(N):
        # 1. Apply tanh activation on previous outputs
        xi1 = np.tanh(y_prev1)  # Activation y(k-1)
        xi2 = np.tanh(y_prev2)  # Activation y(k-2)
        xi3 = x[k]              # Current noisy input

        # 2. Predict current output using RHONN high-order terms
        y[k] = (w[0] * xi1 +
                w[1] * xi1 * xi2 +
                w[2] * xi1 * xi3 +
                w[3] * xi2 * xi3 +
                w[4] * xi3)

        # 3. Compute prediction error
        e = gt[k] - y[k]

        # 4. Construct Jacobian vector for EKF weight update
        H = np.array([
            xi1,
            xi1 * xi2,
            xi1 * xi3,
            xi2 * xi3,
            xi3
        ])

        # 5. EKF weight update
        K = P @ H / (R + H.T @ P @ H)          # Kalman gain
        w = w + 0.9 * (K * e)                  # Update weights
        P = (np.eye(n1) - np.outer(K, H)) @ P + Q * np.eye(n1)  # Update covariance

        # 6. Update history
        y_prev2 = y_prev1
        y_prev1 = y[k]

    # Compute and accumulate MSE for the current channel
    mse_valor = np.mean((gt - y) ** 2)
    msetotal += mse_valor
    print(f"Channel {i+1} MSE: {mse_valor:.6f}")

# -------------------- Save final trained weights --------------------
np.save('w.npy', w)  # Save weights as a NumPy binary file
print("Training completed. Final weights saved.")
print(f"Global average MSE: {msetotal / num_columnas:.6f}")
