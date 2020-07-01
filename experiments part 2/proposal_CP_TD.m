function [K_prop, R_prop, log_prior_ratio, log_prop_ratio] = proposal_CP_TD(K,...
    R, prior_params, prop_params)

% [K_prop, R_prop, log_prior_ratio, log_prop_ratio] = proposal_CP_TD(K,...
%     R, prior_params, prop_params)
% 
% This function performs the proposal mehnanism for the particle MCMC
% algorithm for Bayesian model selection. The models in consideration are
% PARAFAC (CP) and Tucker decomposition (TD).
%
% K, R are the current sample values
% prior_probs are the prior probabilities of CP and TD
% lambda_prior is the parameter of the geometric distribution for the
% cardinalities of the latent allocations
% 
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


prior_probs = prior_params.prior_probs;
lambda_priors = prior_params.lambda_priors;

prop_type = prop_params.prop_type;
prob_switch = prop_params.prob_switch;
prop_indep_lambda = prop_params.prop_indep_lambda;
prop_RW_range = prop_params.prop_RW_range;

u = rand < prob_switch;
K_prop = (3 - K)*u + K*(1 - u);
if K == K_prop % same model
    if K == 1 % if CP
        if prop_type == 0
            R_prop = R + (-1)^(rand < 0.5)*randsample(prop_RW_range, 1);
            log_prior_ratio = (R_prop - R)*log(1 - lambda_priors{1});
            log_prop_ratio = 0;
        else
            R_prop = 1 + geornd(prop_indep_lambda{1});
            log_prior_ratio = (R_prop - R)*log(1 - lambda_priors{1});
            log_prop_ratio = -(R_prop - R)*log(1 - prop_indep_lambda{1});
        end
    else % if TD
        R_prop = R;
        d = randsample(3, 1);
        if prop_type == 0
            R_prop(d) = R(d) + (-1)^(rand < 0.5)*randsample(prop_RW_range, 1);
            log_prior_ratio = (R_prop(d) - R(d))*log(1 - lambda_priors{2}(d));
            log_prop_ratio = 0;
        else
            R_prop(d) = 1 + geornd(prop_indep_lambda{2}(d));
            log_prior_ratio = (R_prop(d) - R(d))*log(1 - lambda_priors{2}(d));
            log_prop_ratio = -(R_prop(d) - R(d))*log(1 - prop_indep_lambda{2}(d));
        end
    end
else % move to the other model
    d = randsample(3, 1);
    d_rest = 1:3 ~= d;
        
    if K == 1 % CP to TD
        % Randomly select a dimension and propose
        R_prop(d) = R + (-1)^(rand < 0.5);  % random walk update with range 1
        R_prop(d_rest) = 1 + geornd(prop_indep_lambda{2}(d_rest)); % the other dimensions
        
        % log proposal ratio
        log_prop_ratio = - sum(log(prop_indep_lambda{2}(d_rest)) ...
            + (R_prop(d_rest)-1).*log(1 - prop_indep_lambda{2}(d_rest)));
        
        % log prior ratio
        log_prior_ratio =  -log(lambda_priors{1}) - (R-1)*log(1 - lambda_priors{1}) ...
            + sum(log(lambda_priors{2}) + (R_prop - 1).*log(1 - lambda_priors{2}));
        
    else % TD to CP
        % Randomly select a dimension and proposa a random walk based on
        % its value
        R_prop = R(d) + (-1)^(rand < 0.5);
        
        log_prop_ratio = sum(log(prop_indep_lambda{2}(d_rest)) ...
            + (R(d_rest)-1).*log(1 - prop_indep_lambda{2}(d_rest)));
        
        log_prior_ratio = log(lambda_priors{1}) + (R_prop-1)*log(lambda_priors{1}) ...
            - sum(log(lambda_priors{2}) + (R-1).*log(1 - lambda_priors{2}));

    end
    % add the ratio of prior model probabilities
    log_prior_ratio = log_prior_ratio  + log(prior_probs(K_prop)) - log(prior_probs(K));
end
