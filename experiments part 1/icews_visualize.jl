include("./src/Viz_Logs.jl");
using .Viz_Logs
import PyCall.pyimport, JSON.parse
plt = pyimport("matplotlib.pyplot")

file_path = ARGS[1]
res_folders = split(file_path, "/")
if length(res_folders) == 3
    _, exp_folder, exp_name = res_folders
    title = exp_folder *"/" * replace(exp_name, ".json"=>"")
    img_folder = "img/" * exp_folder
else
    _, consolidated, exp_folder, exp_name = res_folders
    title = consolidated * "/" * exp_folder *"/" * replace(exp_name, ".json"=>"")
    img_folder = "img/" * consolidated * "/" * exp_folder
end
try mkdir(img_folder) catch end
results = parse(open(file_path, "r"));
fontsize=14
plt.ioff()
_, _ = plot_comparison(results["num_results"], results["Rs"][1]:results["Rs"][end], title, fontsize; save=true, format="pdf");
plt.close()

println("Results plotted at " * "./img/" * title * ".pdf")
