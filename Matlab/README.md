# MATLAB code for the experiments regarding model selection

Experiments regarding model selection can be repeated by running the script called main_CP_TD.m

1) For model selection between CP and TD, the critical input lines should be set as follows:

prior_params.prior_probs = [0.5 0.5]; % prior model probabilities
prop_params.prob_switch = 0.5; % make this 0 to stay in the same model
K0 = 2; R0 = [2 2 2]; a0 = 10; % Start from the TD model
N_vec = [200 200];

2) For model selection within PARAFAC (CP), the critical input lines should be set as follows:

prior_params.prior_probs = [1 0]; % prior model probabilities
prop_params.prob_switch = 0; % make this 0 to stay in the same model
K0 = 1; R0 = 10; a0 = 10; % Start from the CP model
N_vec = [500 0];

3) For model selection within TD, the critical inputs input lines should be set as follows:

prior_params.prior_probs = [0 1]; % prior model probabilities
prop_params.prob_switch = 0; % make this 0 to stay in the same model
K0 = 2; R0 = [2 2 2]; a0 = 10; % Start from the TD model
N_vec = [0 200];
