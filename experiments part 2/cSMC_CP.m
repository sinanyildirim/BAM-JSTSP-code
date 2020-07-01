function [log_Z, s_new_path] = cSMC_CP(X, X_tensor, s_path, dim_X, N, a, R, res_per)

% [log_Z, s_path] = SMC_CP(X, X_tensor, dim_X, N, a, R, res_per)
% 
% This function implements the conditional SMC algorithm for the PARAFAC (CP) model.
% 
% Inputs: X and X_tensor are the two different format of the same count data
% and dim_X has the dimensions of X_tensor.
% s_path the is the trajectory of the latent allocations to be conditioned on
% N is the number of particles, 
% a is the hyperparameter. R is the cardinality of the latent variable.
% res_per is the resampling period.
% 
% Outputs: log_Z is the logarithm of the estimate of the marginal
% likelihood of the data in X.
% s_path is a trajectory of latent allocations drawn from the posterior
% distribution.
% 
% Sinan Yıldırım
% Last update: 1 July 2020.



I = dim_X(1);
J = dim_X(2);
K = dim_X(3);

T = size(X, 1);

% objective bayesian estimates
b = a/T;

% Initiate particles
S_IR(1:I) = {(a/(R*I))*ones(R, N)};
S_JR(1:J) = {(a/(R*J))*ones(R, N)};
S_KR(1:K) = {(a/(R*K))*ones(R, N)};
S_R = (a/R)*ones(R, N);

log_w = zeros(1, N);
log_Z = a*log(b) - (a+T)*log(b + 1) - sum(gammaln(X_tensor(:) + 1));

R_samp = zeros(N, T);
Res_ind = zeros(N, T);

for t = 1:T
    % Get the tuple
    i = X(t, 1); j = X(t, 2); k = X(t, 3);
    
    % propapagate particle    
    log_q_mtx = log(S_IR{i}) + log(S_JR{j}) + log(S_KR{k}) - 2*log(S_R);
    
    % calculate the log-sum
    log_w_inc = log_sum_exp(log_q_mtx, 1);
    
    % sample the latent variable(s)
    q_mtx_cumsum = cumsum(exp(log_q_mtx - log_w_inc), 1);
    
    u = rand(1, N);
    c = sum(u < q_mtx_cumsum, 1);
    r_vec = R - c + 1;
    % fix the first sample to the element of the path
    r_vec(1) = s_path(t);
        
    R_samp(:, t) = r_vec;
    
    temp_ind = R*(0:N-1) + r_vec;
    
    S_IR{i}(temp_ind) = S_IR{i}(temp_ind) + 1;
    S_JR{j}(temp_ind) = S_JR{j}(temp_ind) + 1;
    S_KR{k}(temp_ind) = S_KR{k}(temp_ind) + 1;
    S_R(temp_ind) = S_R(temp_ind) + 1;
    
    % update the particle weights
    log_w = log_w + log_w_inc;
    
    
    %%% resample particles
    if mod(t, res_per) == 0 || t == T
        % normalise the particles
        temp_max = max(log_w);
        log_w_sum = log(sum(exp(log_w  - temp_max))) + temp_max;
        log_Z = log_Z + log_w_sum - log(N);
        
        w_norm = exp(log_w - log_w_sum);
        [res_ind] = cond_resample(w_norm, 'multinomial');
        
        for i = 1:I
            S_IR{i} = S_IR{i}(:, res_ind);
        end
        for j = 1:J        
            S_JR{j} = S_JR{j}(:, res_ind);
        end
        for k = 1:K
            S_KR{k} = S_KR{k}(:, res_ind);
        end
        S_R = S_R(:, res_ind);
        
        % reinitialize the particle weights
        log_w = zeros(1, N); 
        
        Res_ind(:, t) = res_ind;
    else
        Res_ind(:, t) = 1:N;
    end
end

bt = randsample(N, 1, true, w_norm);
s_new_path(T) = R_samp(bt, T);
for t = T-1:-1:1
    bt = Res_ind(bt, t);
    s_new_path(t) = R_samp(bt, t);
end
