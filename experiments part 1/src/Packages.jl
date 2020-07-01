using Pkg
packages = Dict{String,Union{Nothing, VersionNumber}}("CSV"=>v"0.5.12","ForwardDiff"=>v"0.10.10","Distributions"=>v"0.19.2","Atom"=>v"0.12.14","Debugger"=>v"0.6.4","JLD"=>v"0.9.2",
"Optim"=>v"0.20.1","Juno"=>v"0.8.2","BayesOpt"=>v"0.1.0","ReverseDiff"=>v"1.2.0","ScikitLearn"=>v"0.5.1","PyCall"=>v"1.91.2","ConjugatePriors"=>v"0.4.0","NPZ"=>v"0.4.0","
Combinatorics"=>v"1.0.0","JSON"=>v"0.21.0","StatsBase"=>v"0.32.2","DataStructures"=>v"0.17.0","IJulia"=>v"1.20.0","Flux"=>v"0.10.3","Plots"=>v"0.29.5","PyPlot"=>v"2.8.2",
"Einsum"=>v"0.4.1","NMF"=>v"0.4.0","Clustering"=>v"0.13.3","GaussianMixtures"=>v"0.3.0","StatsFuns"=>v"0.9.5","FileIO"=>v"1.2.2","InformationMeasures"=>v"0.3.0",
"ImageView"=>v"0.10.8","DataFrames"=>v"0.19.4","SpecialFunctions"=>v"0.8.0")
for key in keys(packages) 
    Pkg.add(Pkg.PackageSpec(;name=key, version=packages[key]))
end

