# MATLAB code for the experiments regarding model selection

Experiments regarding model selection can be repeated by running the script called main_CP_TD.m
The output file is saved in the directory Outputfiles.

Lines between 26 and 46 are where the algorithm parameters are set. 
The following specifications are required depending on the experiment.

1) For model selection between CP and TD, the critical input lines should be set as follows:

prior_params.prior_probs = [0.5 0.5];
prop_params.prob_switch = 0.5;
K0 = 2; R0 = [2 2 2]; a0 = 10;
N_vec = [200 200];

2) For model selection within PARAFAC (CP), the critical input lines should be set as follows:

prior_params.prior_probs = [1 0];
prop_params.prob_switch = 0;
K0 = 1; R0 = 10; a0 = 10;
N_vec = [500 0];

3) For model selection within TD, the critical inputs input lines should be set as follows:

prior_params.prior_probs = [0 1];
prop_params.prob_switch = 0;
K0 = 2; R0 = [2 2 2]; a0 = 10;
N_vec = [0 200];
