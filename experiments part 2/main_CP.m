% This is the main code for running the pMCMC algorithms for model
% selection. The models in consideration are PARAFAC, or shortly the CP
% model, and the Tucker decomposition model.
%
% Sinan Yıldırım
% Last update: 1 July 2020

%% Clear variables and set the seed
clc; clear; close all; fc = 0;

rng_no = 1;
rng(rng_no);

%% Prepare data
load('X_Q99');
T = size(X, 1);

% Prepare the tensor
X_tensor = zeros(dim_X);
for t = 1:T
    X_tensor(X(t, 1), X(t, 2), X(t, 3)) = X_tensor(X(t, 1), X(t, 2), X(t, 3))+1;
end

%% pMCMC for model selection within CP, within TD, or between CP and TD
% prior parameters
prior_params.lambda_priors = {0.3, [0.3, 0.3, 0.3]}; % prior params for card.s
prior_params.prior_probs = [1 0]; % prior model probabilities
prior_params.prior_pow_a = 2; % prior parameter for a ~ 1/a^prior_pow_a

% proposal parameters
prop_params.sigma_q_a = 0.2/sqrt(T); % proposal std for the hyperparmeter
prop_params.prob_switch = 0; % make this 0 to stay in the same model
prop_params.prop_type = 0; % proposal type within model 0: RW, 1: indep. proposa
prop_params.prop_RW_range = 5; % range of the RW within the model
prop_params.prop_indep_lambda = {0.2, [0.2, 0.2, 0.2]}; % indep. prop. parameter

% initial parameters (choose a viable one based on the prior probs)
K0 = 1; R0 = 10; a0 = 10; % Start from the CP model
% K0 = 2; R0 = [2 2 2]; a0 = 10; % Start from the TD model

M = 1000; % number of iterations
res_per = 1; % resampling period
cSMC_update = 1; % determines whether a cSMC update is to be performed
N_vec = [500 200]; % numbers of particles for SMC for CP and TD, resp.
Num_of_exp = 10; % Number of parallel chains
P = 1; % number of parallel SMC systems: In the paper, this is 1.

% Run the algorithm pMCMC
outputs = cell(Num_of_exp, 1);
parfor i = 1:Num_of_exp
    disp(i);
    [outputs{i}] = pMCMC_CP_TD(X, dim_X, M, N_vec, P, K0, R0, a0, res_per, cSMC_update, ...
        prior_params, prop_params);
end

%% save the data
filename = sprintf('Outputfiles/CP_vs_TD_probs_%02d_%02d_N_%d_%d_M_%d_chain_%d_cSMC_%d_P_%d_rng_%d',...
    10*prior_params.prior_probs(1), 10*prior_params.prior_probs(2), N_vec(1),...
    N_vec(2), M, Num_of_exp, cSMC_update, P, rng_no);

save([filename '_' date]);

%% Organise the outputs: Concatenate the last m_conv of samples from the chains
m_conv = min(100, M);
m_burn = M-m_conv+1:M;
K_Samp = zeros(m_conv, Num_of_exp);
R_Samp = zeros(m_conv, Num_of_exp);
R1_Samp = zeros(m_conv, Num_of_exp);
R2_Samp = zeros(m_conv, Num_of_exp);
R3_Samp = zeros(m_conv, Num_of_exp);
A_Samp = zeros(m_conv, Num_of_exp);
log_Z_Samp = zeros(m_conv, Num_of_exp);

for i = 1:Num_of_exp
    K_Samp(:, i) = outputs{i}.K_samp(m_burn);
    R_Samp(:, i) = outputs{i}.R_samp(m_burn);
    R1_Samp(:, i) = outputs{i}.R_samp(m_burn, 1);
    R2_Samp(:, i) = outputs{i}.R_samp(m_burn, 2);
    R3_Samp(:, i) = outputs{i}.R_samp(m_burn, 3);
    A_Samp(:, i) = outputs{i}.a_samp(m_burn);
    log_Z_Samp(:, i) = outputs{i}.log_Z_samp(m_burn);
end

%% If only one model is considered, plot the histograms
% CP (PARAFAC)
if prior_params.prior_probs(1) == 1
    set(0,'DefaultAxesTitleFontWeight','normal');
    subplot(1, 3, 1);
    histogram(R_Samp(:), 'Normalization', 'probability');
    xlabel('R');
    title('Estimated posterior for R');
    subplot(1, 3, 2);
    histogram(A_Samp(:), 20,  'Normalization', 'probability');
    xlabel('a');
    title('Estimated posterior for a');
    subplot(1, 3, 3);
    histogram(log_Z_Samp(:), 20,  'Normalization', 'probability');
    xlabel('$\hat{\mathcal{L}}_{X}(a, R)$', 'Interpreter', 'LaTex');
    title('Samples for the log-lkl');
end

% TD
if prior_params.prior_probs(2) == 1
    subplot(2, 3, 1);
    histogram(R1_Samp(:), 'Normalization', 'probability');
    xlabel('R_1');
    title('Estimated posterior for R_1');
    subplot(2, 3, 2);
    histogram(R2_Samp(:), 'Normalization', 'probability');
    xlabel('R_2');
    title('Estimated posterior for R_2');
    subplot(2, 3, 3);
    histogram(R3_Samp(:), 'Normalization', 'probability');
    xlabel('R_3');
    title('Estimated posterior for R_3');
    subplot(2, 3, 4);
    histogram(A_Samp(:), 20, 'Normalization', 'probability');
    xlabel('a');
    title('Estimated posterior for a');
    subplot(2, 3, 5);
    histogram(log_Z_Samp(:), 20,  'Normalization', 'probability');
    xlabel('a');
    title('Estimated posterior for log_Z');
end
