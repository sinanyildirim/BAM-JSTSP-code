function a_new = update_a_TD(X, dim_X, a, s, prior_pow_a, sigma_q_a, R)

% a_new = update_a_TD(X, dim_X, s, prior_pow_a, sigma_q_a, R)
%
% This function performs a Metropolis-Hastings update for a for the Tucker
% decomposition model.
% 
% Written by: Sinan Yıldırım
% Last update: 15 June 2020

R1 = R(1);
R2 = R(2);
R3 = R(3);

D1 = dim_X(1);
D2 = dim_X(2);
D3 = dim_X(3);

% Proposal
a_prop = exp(log(a) + sigma_q_a*randn);

T = size(X, 1);

% objective bayesian estimates
b = a/T;
b_prop = a_prop/T;

R_prod = R1*R2*R3;

a1 = a/(R1*D1);
a2 = a/(R2*D2);
a3 = a/(R3*D3);
a4 = a/R_prod;
ar1 = a/R1;
ar2 = a/R2;
ar3 = a/R3;

a1_prop = a_prop/(R1*D1);
a2_prop = a_prop/(R2*D2);
a3_prop = a_prop/(R3*D3);
a4_prop = a_prop/R_prod;
ar1_prop = a_prop/R1;
ar2_prop = a_prop/R2;
ar3_prop = a_prop/R3;

% Initiate the sufficient statistics

S_IR = zeros(D1, R1);
S_JR = zeros(D2, R2);
S_KR = zeros(D3, R3);
S_R1 = zeros(1, R1);
S_R2 = zeros(1, R2);
S_R3 = zeros(1, R3);
S_R = zeros(R1, R2, R3);


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

r1_vec =  RRR(s, 1);
r2_vec =  RRR(s, 2);
r3_vec =  RRR(s, 3);
   
log_Z = a*log(b) - (a+T)*log(b + 1);
log_Z_prop = a_prop*log(b_prop) - (a_prop+T)*log(b_prop + 1);

for t = 1:T
    % Get the tuple
    i = X(t, 1); j = X(t, 2); k = X(t, 3); 
    r1 = r1_vec(t); r2 = r2_vec(t); r3 = r3_vec(t);
    
    % propapagate particle
    log_Z = log_Z + log(S_R(r1, r2, r3) + a4)...
        + log(S_IR(i, r1) + a1) - log(S_R1(r1) + ar1) ...
        + log(S_JR(j, r2) + a2) - log(S_R2(r2) + ar2) ... 
        + log(S_KR(k, r3) + a3) - log(S_R3(r3) + ar3);
  
    log_Z_prop = log_Z_prop + log(S_R(r1, r2, r3) + a4_prop)...
        + log(S_IR(i, r1) + a1_prop) - log(S_R1(r1) + ar1_prop) ...
        + log(S_JR(j, r2) + a2_prop) - log(S_R2(r2) + ar2_prop) ... 
        + log(S_KR(k, r3) + a3_prop) - log(S_R3(r3) + ar3_prop);
    
    % update the sufficient statistics
    S_IR(i, r1) = S_IR(i, r1) + 1;
    S_JR(j, r2) = S_JR(j, r2) + 1;
    S_KR(k, r3) = S_KR(k, r3) + 1;
    S_R(r1, r2, r3) = S_R(r1, r2, r3) + 1;
    S_R1(r1) = S_R1(r1) + 1;
    S_R2(r2) = S_R2(r2) + 1;
    S_R3(r3) = S_R3(r3) + 1;
end

log_ar = (log_Z_prop - log_Z) + (log(a_prop) - log(a))*(prior_pow_a-1);

decision = rand < exp(log_ar);
a_new = decision*a_prop + (1 - decision)*a;
