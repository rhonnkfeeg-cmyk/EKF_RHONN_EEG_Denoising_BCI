clc; clear all; close all;
%% ------------------------------------------------------------------------
%  RHONN EEG Denoising (Inference) Script - Automated Version
%
%  Description:
%  This script performs EEG signal denoising using previously trained RHONN 
%  weights with an Extended Kalman Filter (EKF) approach. It processes 
%  EEG signals channel by channel, applies the high-order recurrent network 
%  inference, and stores the denoised output.
%
%  Main Functions:
%    1. Creates a results folder 'RHONN_results' if it does not exist.
%    2. Allows selection of dataset and contamination type:
%       - dataset_option: noise level ('original', '30pct', '50pct', '70pct', '90pct').
%       - contamination_type: type of contamination ('eeg_emg', 'eeg_eog', 'eeg_eog_emg', 'eeg_heart').
%    3. Loads EEG ground truth and noisy test signals based on selection.
%    4. Iterates through each EEG channel:
%       a. Loads trained RHONN weights.
%       b. Applies high-order recurrent network inference with EKF.
%       c. Stores denoised signal for each channel.
%    5. Saves all denoised EEG signals in MATLAB (.mat) format.
%
%  Output:
%    - MATLAB file 'Deniosing_data_RHONN.mat' containing all denoised EEG channels.
%
%  Authors: Martínez Madrid, Enrique 
%           Medrano Hermosillo, Jesús Alfonso
%           Ramírez Quintana, Juan Alberto
%           Rodríguez Mata, Abraham Efraim
%           González Huitrón, Victor Alejandro
%           Urbina Leos, Iván Ramón
% ------------------------------------------------------------------------

%% ---------------- Create results folder ----------------
folder_name = 'RHONN_results';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

%% ---------------- Select dataset ----------------
% =====================================================
% IMPORTANT:
% Use the SAME 'dataset_option' and 'contamination_type'
% here and in the visualization script (graf_results_optimized.m).
% If they are different, signals and MSE results will not match.
% =====================================================
% Options: 'original', '30pct', '50pct', '70pct', '90pct'
dataset_option = 'original';  % Change according to the dataset you want to process

% Contamination types: 'eeg_emg', 'eeg_eog', 'eeg_eog_emg', 'eeg_heart'
contamination_type = 'eeg_emg';

%% ---------------- Load data based on selection ----------------
switch dataset_option
    case 'original'
        EEG_all_GT_file  = 'data/preprocesing_data/EEG_all_GT_norma_test.mat';
        EEG_noise_file   = 'data/preprocesing_data/EEG_noisy_norma_test.mat';
    case {'30pct','50pct','70pct','90pct'}
        base_path = ['data/real_noisy__data/P_' dataset_option '/' contamination_type '/FormatoMatlab/'];
        EEG_all_GT_file = [base_path 'EEG_all_GT_norma_test.mat'];
        EEG_noise_file  = [base_path 'EEG_noisy_norma_test.mat'];
    otherwise
        error('Dataset option not recognized');
end

load(EEG_all_GT_file);
load(EEG_noise_file);

EEG_all_GT = EEG_all_GT_norma_test;
EEG_noise  = EEG_noisy_norma_test;

%% ---------------- Initialize ----------------
num_columnas = size(EEG_all_GT, 2);
Deniosing_data_RHONN = zeros(size(EEG_all_GT));

%% ---------------- Loop through EEG channels ----------------
for i = 1:num_columnas
    
    % Load trained weights for RHONN
    load('w.mat') 
    
    x = EEG_noise(:, i);  
    N = length(x);        
    
    y = zeros(1, N);      
    y_prev1 = 0;  
    y_prev2 = 0;  
    
    for k = 1:N
        xi1 = tanh(y_prev1);
        xi2 = tanh(y_prev2);
        xi3 = x(k);
             
        y(k) =  w(1)*xi1 + ...
                w(2)*xi1*xi2 + ...
                w(3)*xi1*xi3 + ...
                w(4)*xi2*xi3 + ...
                w(5)*xi3;
        
        y_prev2 = y_prev1;
        y_prev1 = y(k);
    end

    Deniosing_data_RHONN(:, i) = y;
    fprintf('Signal %d processed and stored\n', i);
end

%% ---------------- Save denoised EEG signals ----------------
output_file = fullfile(folder_name, 'Deniosing_data_RHONN.mat');
save(output_file, 'Deniosing_data_RHONN');
fprintf('All denoised EEG signals saved successfully to %s.\n', output_file);
