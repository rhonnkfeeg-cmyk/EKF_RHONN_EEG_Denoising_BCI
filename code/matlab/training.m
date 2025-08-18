%% ------------------------------------------------------------------------
%  RHONN EEG Denoising Training Script
%  This script implements the Recurrent High-Order Neural Network (RHONN)
%  with an Extended Kalman Filter (EKF) for training on EEG signals with
%  additive Gaussian noise, based on methods described in
%         
%  Denoising of EEG Signals in Brain–Computer Interfaces Using 
%  Kalman-Filtered High-Order Recurrent Neural Networks.
%
%  Main steps:
%    1. Load clean EEG signals and their noisy versions.
%    2. Initialize network weights and EKF parameters.
%    3. Loop through each EEG channel:
%       a. Predict the signal using a high-order recurrent neural network.
%       b. Compute the prediction error with respect to ground truth.
%       c. Update weights using the EKF algorithm.
%       d. Track and accumulate mean squared error (MSE).
%    4. Save final trained weights to file.
%
%  Authors: Martínez Madrid, Enrique 
%           Medrano Hermosillo, Jesús Alfonso
%           Ramírez Quintana, Juan Alberto
%           Rodríguez Mata, Abraham Efraim
%           González Huitrón, Victor Alejando
%           Urbina Leos, Iván Ramón
%  ------------------------------------------------------------------------

clc; clear all; close all;

%% ---------------- Load training data ----------------
% Load normalized ground truth and noisy EEG signals
load('data\preprocesing_data\EEG_all_GT_norma_train.mat');
load('data\preprocesing_data\EEG_noisy_norma_train.mat');

EEG_all_GT = EEG_all_GT_norma_train;  % Ground truth EEG signals
EEG_noise  = EEG_noisy_norma_train;   % Noisy EEG signals

num_columnas = size(EEG_all_GT, 2);   % Number of EEG channels

%% ---------------- Initialize RHONN and EKF parameters ----------------
n1 = 5;                  % Number of weights in the network
P = 10 * eye(n1);        % Initial covariance matrix
R = 100;                 % Measurement noise variance
Q = 1e-6;                % Process noise for EKF
H = zeros(n1,1);         % Jacobian placeholder
w = rand(1,n1);          % Random initial weights
msetotal = 0;            % Initialize total MSE

%% ---------------- Loop through EEG channels ----------------
for i = 1:num_columnas
    
    gt = EEG_all_GT(:, i);   % Ground truth signal for current channel
    x  = EEG_noise(:, i);    % Noisy signal for current channel
    N  = length(x);          % Signal length

    % Initialize network output and history
    y = zeros(1, N);         
    y_prev1 = 0;  % y(k-1)
    y_prev2 = 0;  % y(k-2)
     
    %% ---------------- Time-step loop ----------------
    for k = 1:N
        % 1. Apply tanh activation to previous outputs
        xi1 = tanh(y_prev1);  % y(k-1)
        xi2 = tanh(y_prev2);  % y(k-2)
        xi3 = x(k);           % current noisy input
             
        % 2. Predict current output using high-order terms
        y(k) =  w(1)*xi1 + ...
                w(2)*xi1*xi2 + ...
                w(3)*xi1*xi3 + ...
                w(4)*xi2*xi3 + ...
                w(5)*xi3;
                           
        % 3. Compute prediction error
        e = gt(k) - y(k);
        
        % 4. Construct Jacobian vector for EKF
        H = [xi1;
             xi1*xi2;
             xi1*xi3;
             xi2*xi3;
             xi3];

        % 5. EKF weight update
        K = P * H / (R + H' * P * H);  % Kalman gain
        w = w + 0.9 * (K' * e);        % Update weights
        P = (eye(n1) - K * H') * P + Q; % Update covariance

        % 6. Update history
        y_prev2 = y_prev1;
        y_prev1 = y(k);
        
    end
   
    % Compute MSE for the current channel
    mse_valor = mean((gt(:) - y(:)).^2);
    msetotal = msetotal + mse_valor;
    fprintf('MSE Signal %d: %f\n', i, mse_valor);
     
end

%% ---------------- Save final weights ----------------
save('w.mat','w');
fprintf('Training completed. Final weights saved.\n');
fprintf('Global average MSE: %.6f\n', msetotal / num_columnas);


