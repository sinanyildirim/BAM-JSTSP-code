function log_w_norm = log_sum_exp(log_w_mtx, dim_s)

if nargin == 1
    if size(log_w_mtx, 1) == 1
        dim_s = 2;
    else
        dim_s = 1;
    end 
end

temp_max = max(log_w_mtx, [], dim_s);

% calculate the log-sum
log_w_norm = log(sum(exp(log_w_mtx - temp_max), dim_s)) + temp_max;

