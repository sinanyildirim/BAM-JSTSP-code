module Experiments

include("./Misc.jl");
using CSV, JSON, .Misc, Dates

export read_and_convert_icews_data, prepare_params, get_smc_results, get_vb_results, experiment_bam, get_exact_results

function read_and_convert_icews_data(icews_data_file_prefix)
    # reading data file and the associated label files
    df = convert(Matrix{Int64}, CSV.read("$(icews_data_file_prefix).csv", header=false));
    io = open("$(icews_data_file_prefix)_country_dict.json", "r"); country_dict = JSON.parse(io); close(io);
    io = open("$(icews_data_file_prefix)_quadclass_dict.json", "r"); quad_class_dict = JSON.parse(io); close(io);
    # HACK: First two columns (countries) are 0-indexed, but actions are 1-indexed.
    # The line below tackles this problem. This must be handled better in the future.
    df[:, 3] = df[:, 3] .- 1;
    I1, I2, I3 = maximum(df[:,1]) + 1, maximum(df[:,2]) + 1, maximum(df[:,3]) + 1;
    T = size(df)[1];

    X = zeros(Float64, (I1, I2, I3));
    for i in 1:minimum((T, size(df)[1])) # we can use T as a parameter in the future
        # go to the relevant line add 1 to make them all 1-indexed, and increment the respective index in X
        X[(convert(Array{Int, 1}, df[i, :] .+ 1.))...] += 1.
    end
    sequence = [Tuple(df[i,:] .+ 1) for i in 1:size(df)[1]] # make 1-indexed tuples out of each line
    return X, country_dict, quad_class_dict, sequence
end

function prepare_params(X, a)
    μ = nanmean(X)
    μ = μ == 0. ? 0.1 : μ
    γ = a / reduce(*, size(X))
    return μ, γ
end

function get_smc_results(X, Rs, M, N, params, model)
    res = reshape([ret for R=Rs, m=1:M for ret=model.smc_weight(X, R, N; params...)],3,length(Rs),M)
    log_Z, S = Array{Real}(res[1,:,:]), Array{model.Particle}(res[3,:,:])
    inf_results = S[argmax(log_Z, dims=2)]
    log_Z = logsumexp(log_Z, 2) .- log(M)
    return log_Z, inf_results[:]
end

function get_vb_results(X, Rs, M, params, model)
    no_outputs = 4
    res = reshape([ret for R=Rs, m=1:M for ret=model.standard_VB(X, R; params...)],no_outputs,length(Rs),M)
    logW = [res[1,R,m][end] for R=1:length(Rs), m=1:M]
    thetas = [res[no_outputs,R,m] for R=1:length(Rs), m=1:M]
    best_bound = argmax(logW, dims=2)
    log_Z = logW[best_bound]
    inf_results = thetas[best_bound]
    return log_Z[:], inf_results[:]
end

function get_exact_results(X, Rs, params, model)
    return [model.log_marginal(X, R; params...) for R=Rs]
end

function experiment_bam(X::Array; a, Rs, N, M, EPOCHS, adaptive=true, methods=[(CP, "smc"), (CP, "vb")], resampling_freq=1, verbose=true, resampling_scheme="systematic", observation_order=NaN)
    μ, γ = prepare_params(X, a)
    smc_params, vb_params, exact_params, results = Dict(:μ=>μ, :γ=>γ, :adaptive=>adaptive, :resampling_freq=>resampling_freq,
                                                        :resampling_scheme=>resampling_scheme, :observation_order=>observation_order),
                                                        Dict(:μ=>μ, :γ=>γ, :EPOCHS=>EPOCHS), Dict(:γ=>γ), Dict()
    durations = Dict(); durations[Rs] = Dict()
    # We can specify different M's for different methods, or an integer for all to be the same
    M = typeof(M) == Array{Int64,1} ? M : [M for i=1:length(methods)]; @assert (typeof(M) == Array{Int64,1}) && (length(M) == length(methods))
    for (i, method) in enumerate(methods)
        model, inf_method = method
        name = lowercase(replace(string(model, "_", inf_method), "Main."=>"";)); start_time = Dates.now()
        verbose ? print("Starting $(name)... ") : nothing
        if inf_method == "smc"
            log_Z, inf_results = get_smc_results(X, Rs, M[i], N, smc_params, model)
        elseif inf_method == "vb"
            log_Z, inf_results = get_vb_results(X, Rs, M[i], vb_params, model)
        elseif inf_method == "exact"
            log_Z, inf_results = get_exact_results(X, Rs, exact_params, model), [] # no inf_results from exact computation of the likelihood
        else
            throw(ArgumentError("Model not recognized."))
        end
        duration = round((Dates.now() - start_time).value / 10)/100; durations[Rs][method] = duration
        verbose ? println(string("Finished in ", duration," s.")) : nothing
        results[name] = Dict("log_Z"=>log_Z, "inf_results"=>inf_results)
    end
    return results, durations
end

end
