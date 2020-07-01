using JSON
include("./src/Viz_Logs.jl");
using .Viz_Logs

exp_name = "2017_week_0.5_quantile_quad_20_countries_cp"
a = "1.0"
N = "123140" # 3750 17730 35190 78500 123140  d= 0.07 0.20 0.32 0.53 0.67

exp_name = ARGS[1]
a = ARGS[2]
N = ARGS[3]

run_name_rless = "exp_a_$(a)_N_$(N)_EPOCHS_100_f_0_Rs_"
results_folder = "results/$(exp_name)/";

folder_contents = [file for file in readdir(results_folder) if occursin(run_name_rless, file) ]
exp_files = folder_contents[sortperm([parse(Int64, match(r"_[0-9]+:", file).match[2:end-1]) for file in folder_contents])];
results = JSON.parse(open(string(results_folder, exp_files[1]), "r"))
R_min = results["Rs"][1]
R_max = JSON.parse(open(string(results_folder, exp_files[end]), "r"))["Rs"][end]
Rs = R_min:R_max
results["Rs"] = Rs;

methods = keys(results["num_results"])
for method in methods
    results["num_results"][method] = Dict()
    results["num_results"][method]["log_Z"] = vcat([JSON.parse(open(string(results_folder, exp_file), "r"))["num_results"][method]["log_Z"] for exp_file in exp_files]...)
    results["num_results"][method]["inf_results"] = vcat([JSON.parse(open(string(results_folder, exp_file), "r"))["num_results"][method]["inf_results"] for exp_file in exp_files]...)
end;
for exp_file in exp_files
    results_exp = JSON.parse(open(string(results_folder, exp_file), "r"))
    key = string(results_exp["Rs"][1]:results_exp["Rs"][end])
    try results["durations"][key] = results_exp["durations"][key] catch nothing end
end

try
mkdir("results/consolidated")
mkdir("results/consolidated/" * exp_name)
catch
end

results_write_json("consolidated/"*exp_name, results["a"], results["N"], results["M"], results["EPOCHS"], results["Rs"],
        results["resampling_freq"], results["resampling_scheme"],
        results["num_results"], results["adaptive"]; durations=results["durations"])

println("Consolidation completed, see your file inside the folder results/consolidated/"*exp_name*"/")
