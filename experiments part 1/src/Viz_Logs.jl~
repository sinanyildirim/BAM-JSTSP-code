module Viz_Logs

export plot_cp, results_write_json, plot_comparison

import JSON.json, PyCall.pyimport, Dates.now
plt_org = pyimport("matplotlib.pyplot")

function plot_cp(S)
    plt_org.imshow(reshape(S["S_R"], (1, length(S["S_R"]))));  
    
    S_IR = hcat( S["S_IR"]...)
    fig, ax = plt_org.subplots()
    ax.imshow(S_IR./sum(S_IR, dims=1), extent=[-0.5, R - 0.5, 0, 20])
    ax.set_yticks(0.5 .+ 0:1:20);
    ax.set_yticklabels([country_dict[string(19 - i)] for i in 0:19]);
    ax.set_xticks(0:R-1);
    
    S_JR = hcat( S["S_JR"]...)
    fig, ax = plt_org.subplots()
    ax.imshow(S_JR./sum(S_JR, dims=1), extent=[-0.5, R - 0.5, 0, 20])
    ax.set_yticks(0.5 .+ 0:1:20);
    ax.set_yticklabels([country_dict[string(19 - i)] for i in 0:19]);
    ax.set_xticks(0:R-1);
    
    S_KR = hcat( S["S_KR"]...)
    fig, ax = plt_org.subplots()
    ax.imshow(S_KR./sum(S_KR, dims=1), extent=[-0.5, R - 0.5, -0.5, 3.5])
    ax.set_yticks(0.5 .+ 0:1:4);
    ax.set_xticks(0:R-1);
end 

function results_write_json(exp_name, a, N, M, EPOCHS, Rs, resampling_freq, resampling_scheme, num_results, adaptive; durations=Dict())
    folder_name = string("results/", exp_name)
    try mkdir(folder_name) catch end
    file_name = string(folder_name,"/exp", "_a_", a, "_N_", N, "_EPOCHS_", EPOCHS, "_f_", resampling_freq, "_Rs_", Rs, ".json")
    results = Dict()
    results[:a] = a; results[:N] = N; results[:M] = M; results[:EPOCHS] = EPOCHS; results[:Rs] = Rs
    results[:resampling_freq] = resampling_freq; results[:resampling_scheme] = resampling_scheme;
    results[:num_results] = num_results; results[:adaptive] = adaptive; results[:durations] = durations
    results[:datetime] = now()
    results = json(results)
    open(file_name, "w") do f
        write(f, results)
    end;
end;

function screen_name(name::String) return join(split(uppercase(replace(name, "vb"=>"ELBO")), "_"), " ") end

function plot_comparison(results, available_Rs, title; figsize=(5,3), save=false, format="pdf", 
                                                       selected_Rs=available_Rs[1]:available_Rs[end])
    @assert typeof(available_Rs) == UnitRange{Int64} # the provided range must be UnitRange
    s = selected_Rs[1] - available_Rs[1] + 1 # the starting index
    e = selected_Rs[end] - selected_Rs[1] + s # the ending index
    fig, ax = plt_org.subplots(figsize=figsize);
    for name in keys(results)
        log_Z = results[name]["log_Z"]
        @assert length(log_Z) == length(available_Rs) # The original lengths of available Rs and results must match
        ax.plot(available_Rs[s:e], log_Z[s:e], label=screen_name(name), marker="o")
    end
    ax.set_ylabel("\$ \\log \\mathcal{L}_X(R) \$"); ax.set_xlabel("\$R\$ (Model order)");
    ax.set_xticks(selected_Rs[1:2:end]); ax.legend(); ax.set_title(title);
    if save fig.savefig("img/$(title).$(format)", format=format, bbox_inches="tight") end
    return fig, ax
end;         

function plot_comparison(results, available_Rs, title, fontsize; figsize=(5,3), save=false, format="pdf", 
                                                       selected_Rs=available_Rs[1]:available_Rs[end])
    @assert typeof(available_Rs) == UnitRange{Int64} # the provided range must be UnitRange
    s = selected_Rs[1] - available_Rs[1] + 1 # the starting index
    e = selected_Rs[end] - selected_Rs[1] + s # the ending index
    fig, ax = plt_org.subplots(figsize=figsize);
    for name in keys(results)
        log_Z = results[name]["log_Z"]
        @assert length(log_Z) == length(available_Rs) # The original lengths of available Rs and results must match
        ax.plot(available_Rs[s:e], log_Z[s:e], label=screen_name(name), marker="o")
    end
    ax.set_ylabel("\$ \\log \\mathcal{L}_X(R) \$", fontsize=fontsize); ax.set_xlabel("\$R\$ (Model order)", fontsize=fontsize)
    ax.set_xticks(selected_Rs[1:3:end]); ax.legend(fontsize=fontsize-1); #ax.set_title(title, fontsize=fontsize);
    ax.tick_params(axis="both", labelsize=fontsize-1)
    if save fig.savefig("img/$(title).$(format)", format=format, bbox_inches="tight") end
    return fig, ax
end; 

end