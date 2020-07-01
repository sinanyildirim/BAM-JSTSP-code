function a_new = update_a_CP(X, dim_X, a, s, prior_pow_a, sigma_q_a, R)

% a_new = update_a_CP(X, dim_X, s, prior_pow_a, sigma_q_a, R)
%
% This function performs a Metropolis-Hastings update for the hyperparameter a
% for the CP model.
% 
% Written by: Sinan Yıldırım
% Last update: 15 June 2020

D1 = dim_X(1);
D2 = dim_X(2);
D3 = dim_X(3);

% Proposal
a_prop = exp(log(a) + sigma_q_a*randn);

T = size(X, 1);

% objective bayesian estimates
b = a/T;
b_prop = a_prop/T;

a1 = a/(R*D1);
a2 = a/(R*D2);
a3 = a/(R*D3);
a4 = a/R;

a1_prop = a_prop/(R*D1);
a2_prop = a_prop/(R*D2);
a3_prop = a_prop/(R*D3);
a4_prop = a_prop/R;

% Initiate the sufficient statistics
S_IR = zeros(D1, R);
S_JR = zeros(D2, R);
S_KR = zeros(D3, R);
S_R = zeros(1, R);

log_Z = a*log(b) - (a+T)*log(b + 1);
log_Z_prop = a_prop*log(b_prop) - (a_prop+T)*log(b_prop + 1);


for t = 1:T
    % Get the tuple
    i = X(t, 1); j = X(t, 2); k = X(t, 3); r = s(t);
    
    % propapagate particle
    log_Z = log_Z + log(S_IR(i, r)+a1) + log(S_JR(j, r)+a2) + log(S_KR(k, r)+a3) ...
        - 2*log(S_R(r)+a4);
    
    log_Z_prop = log_Z_prop + log(S_IR(i, r)+a1_prop) + log(S_JR(j, r)+a2_prop) ...
        + log(S_KR(k, r)+a3_prop) - 2*log(S_R(r)+a4_prop);
    
    % update the sufficient statistics
    S_IR(i, r) = S_IR(i, r) + 1;
    S_JR(j, r) = S_JR(j, r) + 1;
    S_KR(k, r) = S_KR(k, r) + 1;
    S_R(r) = S_R(r) + 1;
    
end

log_ar = log_Z_prop - log_Z + (log(a_prop) - log(a))*(prior_pow_a-1);

decision = rand < exp(log_ar);
a_new = decision*a_prop + (1 - decision)*a;
