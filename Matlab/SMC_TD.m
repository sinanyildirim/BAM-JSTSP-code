function [log_Z, s_path_new] = SMC_TD(X, X_tensor, dim_X, N, a, R_vec, res_per)

% [log_Z, s_path, particles] = SMC_TD(X, X_tensor, dim_X, N, a, R_vec, res_per)
% 
% This function implements the SMC algorithm for Tucker decomposition.
% 
% Inputs: X and X_tensor are the two different format of the same count data
% and dim_X has the dimensions of X_tensor. N is the number of particles, 
% a is the hyperparameter. R_vec has the dimensions of the latent variable.
% res_per is the resampling period.
% 
% Outputs: log_Z is the logarithm of the estimate of the marginal
% likelihood of the data in X.
% s_path is a trajectory of latent allocations drawn from the posterior
% distribution.
% 
% Sinan Yıldırım
% Last update: 1 July 2020

R1 = R_vec(1);
R2 = R_vec(2);
R3 = R_vec(3);

I = dim_X(1);
J = dim_X(2);
K = dim_X(3);

T = size(X, 1);

% objective bayesian estimates
b = a/T;

log_w = zeros(1, N);
log_Z = a*log(b) - (a+T)*log(b + 1) - sum(gammaln(X_tensor(:) + 1));

R_prod = R1*R2*R3;
RRR = zeros(R_prod, 3);

ind = 0;
for r1 = 1:R1
    for r2 = 1:R2
        for r3 = 1:R3
            ind = ind + 1;
            RRR(ind, :) = [r1, r2, r3];
        end
    end
end

R1_base = R1*(0:N-1);
R2_base = R2*(0:N-1);
R3_base = R3*(0:N-1);
R_prod_base = R_prod*(0:N-1);

% Initiate particles
S_IR(1:I) = {(a/(R1*I))*ones(R1, N)};
S_JR(1:J) = {(a/(R2*J))*ones(R2, N)};
S_KR(1:K) = {(a/(R3*K))*ones(R3, N)};
S_R = (a/R_prod)*ones(R3, R2, R1, N);

if nargout >= 2
    R_samp = zeros(N, T);
    Res_ind = zeros(N, T);
end

for t = 1:T
    % Get the tuple
    i = X(t, 1); j = X(t, 2); k = X(t, 3);
    
    % Reshape S_R
    S_R1 = reshape(sum(S_R, [1, 2]), R1, N); % R1 x N
    S_R2 = reshape(sum(S_R, [1, 3]), R2, N); % R2 x N
    S_R3 = reshape(sum(S_R, [2, 3]), R3, N); % R3 x N
    
    % Calculate the log proposal probabilities
    log_q_mtx1 = log(S_R) + reshape(log(S_IR{i}) - log(S_R1), 1, 1, R1, N) ...
        + reshape(log(S_JR{j}) - log(S_R2), 1, R2, 1, N)...
        + reshape(log(S_KR{k}) - log(S_R3), R3, 1, 1, N);
    
    log_q_mtx = reshape(log_q_mtx1, R_prod, N);
    
    % calculate the log-sum of the proposal probabilities
    log_w_inc = log_sum_exp(log_q_mtx, 1);
    
    % sample the latent variable(s)
    q_mtx_cumsum = cumsum(exp(log_q_mtx - log_w_inc), 1);
    u = rand(1, N);
    c = sum(u < q_mtx_cumsum, 1);
    r_vec = R_prod - c + 1;
    
    r1_vec =  RRR(r_vec, 1)';
    r2_vec =  RRR(r_vec, 2)';
    r3_vec =  RRR(r_vec, 3)';
    
    temp_ind_1 = R1_base + r1_vec;
    temp_ind_2 = R2_base + r2_vec;
    temp_ind_3 = R3_base + r3_vec;
    temp_ind = R_prod_base + r_vec;
    
    S_IR{i}(temp_ind_1) = S_IR{i}(temp_ind_1) + 1;
    S_JR{j}(temp_ind_2) = S_JR{j}(temp_ind_2) + 1;
    S_KR{k}(temp_ind_3) = S_KR{k}(temp_ind_3) + 1;
    S_R(temp_ind) = S_R(temp_ind) + 1;
    
    % update the particle weights
    log_w = log_w + log_w_inc;
    
    if nargout >= 2
        R_samp(:, t) = r_vec;
    end
    
    % update the estimates and resample particles
    if mod(t, res_per) == 0 || t == T
        % normalise the particles
        log_w_sum = log_sum_exp(log_w);
        % update the log-lkl estimate
        log_Z = log_Z + log_w_sum - log(N);
        
        % resample
        w_norm = exp(log_w - log_w_sum);
        if t <= T
            [res_ind] = resample(w_norm, 'multinomial');
            for i = 1:I
                S_IR{i} = S_IR{i}(:, res_ind);
            end
            for j = 1:J
                S_JR{j} = S_JR{j}(:, res_ind);
            end
            for k = 1:K
                S_KR{k} = S_KR{k}(:, res_ind);
            end
            S_R = S_R(:, :, :, res_ind);
            
            % reinitialize the particle weights
            log_w = zeros(1, N);
            
            % Keep the ancestory lineage
            if nargout == 2
                Res_ind(:, t) = res_ind;
            end
        end
    else
        % if no resampling, carry out the particles to the next iteration
        if nargout == 2
            Res_ind(:, t) = 1:N;
        end
    end
        
end

% sample the trajectory of latent allocations
if nargout == 2
    bt = randsample(N, 1, true, w_norm);
    s_path_new(T) = R_samp(bt, T);
    
    for t = T-1:-1:1
        bt = Res_ind(bt, t);
        s_path_new(t) = R_samp(bt, t);
    end
end