println("Loaded the experiment file.")

using Random
Random.seed!(42);

include("./src/Misc.jl");
include("./src/CP/CP.jl");
include("./src/Viz_Logs.jl");
include("./src/Experiments.jl");
include("./src/TD/TD.jl");
using .Viz_Logs, .Experiments, .Misc, .CP, .TD
using FileIO, Dates, JSON, CSV, LinearAlgebra, NPZ, Distributions, Plots
using Base.Iterators: flatten

data_name = "icews_0.99_data" # choices are icews_{}_data where {} can take 0.99, 0.95, 0.9, 0.75, 0.5
exp_type = "cp_comparison" # this is up to the user
exp_name = "$(data_name)_$(exp_type)"
icews_data_file_prefix = "data/icews/$(data_name)"
meths = [(CP, "smc"), (CP, "vb")] # choices should be made from (CP, "smc"), (CP, "vb"), (TD, "smc"), and (TD, "vb")
M = [10, 10] # length of M should be equal to that of meths
adaptive = true # choicese are: true, false
# Resampling is default, if you want to prevent resampling, set resampling freq to T
resampling_freq = 20 # choices range from 1 to total number of observations
resampling_freq = adaptive ? 0 : resampling_freq
resampling_scheme = "systematic" # chocies are: "systematic", "multinomial"

X, country_dict, quad_class_dict, sequence = read_and_convert_icews_data(icews_data_file_prefix);

println(string("Experiment name: ", exp_name))
println(string("Adaptive: ", adaptive))
adaptive ? nothing : println(string("Resampling frequency: ", resampling_freq))

a_temp, N_temp, M_temp, EPOCHS_temp, Rs_temp = 1, 1, 1, 1, 1:1
sample_X = zeros(Tuple(2 for i in 1:length(size(X))))
sample_X[Tuple(1 for i in 1:length(size(X)))...] = 1

num_results, durations = experiment_bam(sample_X; a=a_temp, Rs=Rs_temp, N=N_temp, M=M_temp, EPOCHS=EPOCHS_temp,
                            adaptive=adaptive, methods=meths, resampling_freq=resampling_freq,
                            resampling_scheme=resampling_scheme, verbose=false)

args = [i for i in ARGS] # for debugging with IDE
a = parse(Float64, args[1])
N = parse(Int64, args[2])
EPOCHS = parse(Int64, args[3])
Rs = parse(Int64, args[4]):parse(Int64, args[5]);

num_results, durations = experiment_bam(X; a=a, Rs=Rs, N=N, M=M, EPOCHS=EPOCHS, adaptive=adaptive, methods=meths,
                            resampling_freq=resampling_freq, resampling_scheme=resampling_scheme)
results_write_json(exp_name, a, N, M, EPOCHS, Rs, resampling_freq, resampling_scheme, num_results, adaptive; durations=durations)
