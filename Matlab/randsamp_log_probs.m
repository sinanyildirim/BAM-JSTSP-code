function [x, log_norm] = randsamp_log_probs(log_p)

% [ind, log_norm] = randsamp_log_probs(log_w)
% 
% This function returns a random sample from {1, ..., K} from a probabity 
% distribution whose log-(unnormalised) probabilities are given as log_p,
% an array of size K.
% 
% The primary output is the random sample. 
% The second output is the logarithm of the normalising constant for log_p
% 
% Sinan Yıldırım
% Last update: 1 July 2020

N = length(log_p);

log_norm = log_sum_exp(log_p);

p = exp(log_p - log_norm);
x = randsample(N, 1, true, p);