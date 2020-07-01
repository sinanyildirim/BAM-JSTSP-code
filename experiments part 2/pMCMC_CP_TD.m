function [outputs] = pMCMC_CP_TD(X, dim_X, M, N, P, K0, R0, a0, res_per, ...
    cSMC_update, prior_params, prop_params)

% [outputs] = pMCMC_CP_TD(X, dim_X, M, N, P, K0, R0, a0, res_per, ...
%     cSMC_update, prior_params, prop_params)
%
% This function implements an MCMC algorithm for model selection where the
% statistical model is a mixture of tensor factorisation models.
%
% X and dim_X are the sparse representation of a tensor (which is a sequence
% of tuples) and the dimensions of the tensor.
% M is the number of MCMC iterations
% N is a 2x1 vector that contains the number of particles to be used in the
% SMC algorithms run for each model, CP and TD, respectively
% P is the number of parallel SMC systems. (In the paper, P = 1)
% K0, R0, and a0 are the initial parameter for the model number, the
% cardinality vectors, and the hyperparameter.
% res_per is the resampling period for the SMC algorithm
% cSMC update is a binary input that determines whether a conditional cSMC
% move is to be performed to update the sample trajectory for the latent
% allocations as well as the log-likelihood under the current parameter.
% prior_params is a structure that contains
% - prior_pow_a: the prior for a is determined as a ~ 1/a^prior_pow_a
% - prior_probs: this a 2x1 vector of prior probabilities for the models
% - lambda_priors: this is 2x1 cell, where the first element is a scalar and
% the second one is a 3x1 array, containing the success parameters for the
% Geometric prior of R (for CP) or (R(1), R(2), R(3)) (for TD).
% 
% prop_params is a structure that contains the proposal parameters (for the
% details of the proposal mechanism, see the related function proposal_CP_TD
% - prob_switch: This is the probability of proposing the other model
% - prop_type: This determines whether a random walk proposal or an
% independent proposal is to be used for the moves within a model
% - prop_indep_lambda: This is a cell containing the parameters of the 
% geometric distribution that is used to propose the new cardinality R when
% a whitin move is proposed with prop_type being 1 or when a new cardinality
% is to be proposed following a proposal to move to the other model.
% prop_RW_range: This the range of the random walk proposal for the
% cardinality parameter.
% 
% Sinan Yıldırım
% Last update: 1 July 2020


D1 = dim_X(1);
D2 = dim_X(2);
D3 = dim_X(3);

prior_pow_a = prior_params.prior_pow_a;
sigma_q_a = prop_params.sigma_q_a;

model_dims = [1 3];
% prepare the tensor
X_tensor = zeros(D1, D2, D3);
T = size(X, 1);
for t = 1:T
    X_tensor(X(t, 1), X(t, 2), X(t, 3)) = X_tensor(X(t, 1), X(t, 2), X(t, 3)) + 1;
end

K = K0;
R = R0;
a = a0;

fun_names_SMC = {str2func('SMC_CP'), str2func('SMC_TD')};
fun_names_cSMC = {str2func('cSMC_CP'), str2func('cSMC_TD')};
fun_names_update_a = {str2func('update_a_CP'), str2func('update_a_TD')};

% initialize
log_Z_vec = zeros(P, 1);
log_Z_prop_vec = zeros(P, 1);
s_path = cell(P, 1);
s_path_prop = cell(P, 1);

if cSMC_update == 1
    for p = 1:P
        [log_Z_vec(p), s_path{p}] = fun_names_SMC{K}(X, X_tensor, ...
            dim_X, N(K), a, R, res_per);
    end
    % sample a path and calculate the log-lkl estimate
    [p_ind, log_norm_temp] = randsamp_log_probs(log_Z_vec);
    log_Z = log_norm_temp - log(P);
    s_path_samp = s_path{p_ind};
else
    for p = 1:P
        [log_Z_vec(p), ~] = fun_names_SMC{K}(X, X_tensor, dim_X, N(K), a, R, res_per);
    end
    log_Z = log_sum_exp(log_Z_vec) - log(P);
end

K_samp = zeros(1, M);
R_samp = zeros(M, 3);
a_samp = zeros(1, M);
log_Z_samp = zeros(1, M);

% run the algorithm
for m = 1:M
    disp([m, K, R, a log_Z]);
    
    %%% propose and calculate the log-prior and log proposal ratios
    [K_prop, R_prop, log_prior_ratio, log_prop_ratio] = ...
        proposal_CP_TD(K, R, prior_params, prop_params);
    
    % If the cSMC refreshment is not available, then propose a here.
    if cSMC_update == 0
        a_prop = exp(log(a) + sigma_q_a*randn);
        log_prior_ratio = log_prior_ratio - (log(a_prop) - log(a))*(prior_pow_a - 1);
    end
    
    % ratio of log-likelihood
    if sum(R_prop <= 0) == 0
        % run P parallel SMC algorithms at the proposal
        if cSMC_update == 1
            for p = 1:P
                [log_Z_prop_vec(p), s_path_prop{p}] = fun_names_SMC{K_prop}(X, X_tensor, ...
                    dim_X, N(K_prop), a, R_prop, res_per);
            end
            % sample a path and calculate the log-lkl estimate
            [p_ind, log_norm_temp] = randsamp_log_probs(log_Z_prop_vec);
            log_Z_prop = log_norm_temp - log(P);
            s_path_prop_samp = s_path_prop{p_ind};
        else
            for p = 1:P
                [log_Z_prop_vec(p), ~] = fun_names_SMC{K_prop}(X, X_tensor, ...
                    dim_X, N(K_prop), a, R_prop);
            end
            log_Z_prop = log_sum_exp(log_Z_prop_vec) - log(P);
        end
        % ratio of log-likelihoods
        log_lkl_ratio = log_Z_prop - log_Z;
        
        % acceptance ratio
        log_ar = log_prop_ratio + log_prior_ratio + log_lkl_ratio;
        
        % decision and upadte
        decision = rand < exp(log_ar);
        if decision == 1
            % fprintf('I am in %d', m);
            log_Z = log_Z_prop;
            if cSMC_update == 0
                a = a_prop;
            end
            K = K_prop;
            R = R_prop;
            if cSMC_update == 1
                s_path_samp = s_path_prop_samp;
            end
        end
    end
    
    %%% Update the path and the likelihood estimator with cSMC
    if cSMC_update == 1
        % Update a with a MHwG algorithm: This has to go along with cSMC update,
        % otherwise it cannot be used.
        a = fun_names_update_a{K}(X, dim_X, a, s_path_samp, prior_pow_a, sigma_q_a, R);
        for p = 1:P
            if p == 1 % This is cSMC
                [log_Z_vec(p), s_path{p}] = fun_names_cSMC{K}(X, X_tensor,...
                    s_path_samp, dim_X, N(K), a, R, res_per);
            else % The rest should come from SMC
                [log_Z_vec(p), s_path{p}] = fun_names_SMC{K}(X, X_tensor,...
                    dim_X, N(K), a, R, res_per);
            end
        end
        
        % sample a path and calculate the log-lkl estimate
        [p_ind, log_norm_temp] = randsamp_log_probs(log_Z_vec);
        log_Z = log_norm_temp - log(P);
        s_path_samp = s_path{p_ind}; 
    end
    
    % Store the samples
    K_samp(m) = K;
    R_samp(m, 1:model_dims(K)) = R;
    a_samp(m) = a;
    log_Z_samp(m) = log_Z;
end

% return the outputs
outputs.K_samp = K_samp;
outputs.R_samp = R_samp;
outputs.a_samp = a_samp;
outputs.log_Z_samp = log_Z_samp;