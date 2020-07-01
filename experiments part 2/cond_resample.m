function [res_ind] = cond_resample(w, type, rand_perm)

% [res_ind] = cond_resample(w, type, rand_perm)
% 
% This function performs conditional resampling given the weights and the
% type of resampling. It is assumed that the indices (1:N) have weights 
% w(1), w(2), ..., w(N) before resampling and the first resampled indice is
% 1, that is res_ind(1) = 1.
%
% There are three options for "type":
% 1. 'multinomial' performs conditional multinomial resampling
% 2. 'systematic' performs conditional systematic resampling
% 3. 'residual' performs conditional residual resampling
% 
% The conditional systematic and residual resampling is performed as
% described in the paper Chopin and Singh (2013) 'On the Particle Gibbs
% Sampler'.
% 
% Sinan Yildirim, 04.11.2013
% Last update: 05.11.2013, 02.23

% make w a column vector:
w = w(:); w = w/sum(w);
N = length(w);
if nargin == 2
    rand_perm = 'off';
end

if strcmp(type, 'multinomial') == 1
    extra_params.multinom_N = N - 1;
    res_ind = zeros(1, N); 
    res_ind(1) = 1;
    res_ind(2:end) = resample(w, 'multinomial', extra_params);
    if strcmp(rand_perm, 'on') == 1
        res_ind(2:end) = res_ind(randperm(N-1) + 1);
    end
elseif strcmp(type, 'systematic') == 1
    floor_w = floor(N*w);
    r = N*w - floor_w;
    % Step 1: Draw the conditional uniform sample:
    if N*w(1) < 1
        u = N*w(1)*rand;
    elseif rand < r(1)*(floor_w(1) + 1)/(N*w(1))
        u = r(1)*rand;
    else
        u = (1-r(1))*rand + r(1);
    end
    % Step 2: Perform systematic resampling using u
    res_ind = zeros(1, N);
    w_cum = cumsum(w*N);
    j = 1;
    for i = 1:N
        while w_cum(j) < u
        	j = j + 1;
        end
        res_ind(i) = j;
        u = u + 1;
    end
    % Step 3: Random cycling conditional on that the first indice is 1
    if strcmp(rand_perm, 'on') == 1
        c = randgen(0:length(res_ind == 1), 1);
        res_ind = res_ind([c+1:N 1:c]);
    end
elseif strcmp(type, 'residual') == 1
    floor_w = floor(N*w);
    r = N*w - floor_w;
    if rand < floor_w(1)/(N*w(1))
        % res_ind(1) is the deterministic entry of the residual sampling
        res_ind = resample(w, 'residual');
    else
        res_ind = zeros(1, N);
        res_ind(1) = 1;
        ind_cum = 1;
        for i = 1:N
            res_ind(ind_cum+1:ind_cum + floor_w(i)) = i*ones(floor_w(i), 1);
            ind_cum = ind_cum + floor_w(i);
        end
        extra_params.multinom_N = N - ind_cum;
        res_ind(ind_cum+1:end) = resample(r, 'multinomial', extra_params);
    end
    if strcmp(rand_perm, 'on') == 1
        % randomly permute the indices at the entries 2:N
        res_ind(2:end) = res_ind(randperm(N-1) + 1);
    end
end