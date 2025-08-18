clc; clear all; close all;
%% ------------------------------------------------------------------------
%  RHONN EEG Denoising Visualization and MSE Evaluation Script - Automated Version
%
%  Description:
%  This script performs visualization and evaluation of EEG signals denoised 
%  using a pre-trained RHONN model with EKF. It allows comparison of original 
%  (ground truth), noisy, and denoised signals, and computes the Mean Squared 
%  Error (MSE) for each EEG channel.
%
%  Main Functions:
%    1. Creates results folders: 'RHONN_results' and subfolder 'images'.
%    2. Allows automatic selection of dataset and contamination type.
%    3. Loads original, noisy, and denoised EEG signals.
%    4. Iterates over each EEG channel or a limited number of signals:
%       a. Computes MSE between denoised and ground truth signals.
%       b. Plots clean, noisy, and denoised signals.
%       c. Saves plots as PNG images.
%    5. Compiles MSE results into a table, computes average MSE, and saves as CSV.
%
%  Outputs:
%    - PNG images of each EEG channel comparing signals.
%    - CSV file 'MSE_results.csv' with MSE per channel and overall average.
% ------------------------------------------------------------------------

%% ---------------- Create results folders ----------------
folder_name = 'RHONN_results';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

folder_name1 = fullfile(folder_name, 'images');
if ~exist(folder_name1, 'dir')
    mkdir(folder_name1);
end

%% ---------------- Select dataset ----------------
% =====================================================
% IMPORTANT:
% Use the SAME 'dataset_option' and 'contamination_type'
% here and in the inference script (test_optimized.m).
% If they are different, signals and MSE results will not match.
% =====================================================
dataset_option = 'original';      % Options: 'original', '30pct', '50pct', '70pct', '90pct'
contamination_type = 'eeg_emg';   % Options: 'eeg_emg', 'eeg_eog', 'eeg_eog_emg', 'eeg_heart'

%% ---------------- Load test and denoised data ----------------
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

denoised_file = fullfile(folder_name, 'Deniosing_data_RHONN.mat');
load(denoised_file);

%% ---------------- Parameters ----------------
num_columnas = size(EEG_all_GT, 2);
num_signals_to_plot = 10;             
num_signals_to_plot = min(num_signals_to_plot, num_columnas);
mse_lista = zeros(num_columnas, 1);

%% ---------------- Loop through signals to plot ----------------
for i = 1:num_signals_to_plot
    
    gt = EEG_all_GT(:, i);
    x  = EEG_noise(:, i);
    y  = Deniosing_data_RHONN(:, i);
    
    mse_valor = mean((gt(:) - y(:)).^2);
    mse_lista(i) = mse_valor;

    figure;
    plot(gt, 'k', 'LineWidth', 2); hold on;
    plot(x, 'b', 'LineWidth', 1.5); 
    plot(y, 'r', 'LineWidth', 1.5);
    legend('Clean Signal', 'Noisy Signal', 'RHONN + EKF Denoised');
    xlabel('Time'); ylabel('Amplitude');
    title(['EEG Channel ' num2str(i)]);
    set(gcf, 'Position', [100, 100, 1800, 900]);
    grid on;

    saveas(gcf, fullfile(folder_name1, ['plot_signal_' num2str(i) '.png']));
    close(gcf);
    
end

%% ---------------- Compute MSE for all channels ----------------
for i = (num_signals_to_plot+1):num_columnas
    gt = EEG_all_GT(:, i);
    y  = Deniosing_data_RHONN(:, i);
    mse_lista(i) = mean((gt(:) - y(:)).^2);
end

%% ---------------- Create and save MSE results table ----------------
channels = (1:num_columnas)';
tabla_resultados = table(channels, mse_lista, 'VariableNames', {'Signal', 'MSE'});

promedio_mse = mean(mse_lista);
fila_promedio = table(0, promedio_mse, 'VariableNames', {'Signal', 'MSE'});
tabla_resultados = [tabla_resultados; fila_promedio];

writetable(tabla_resultados, fullfile(folder_name, 'MSE_results.csv'));
fprintf('Plots saved and MSE results exported successfully.\n');
