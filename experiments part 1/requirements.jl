import Pkg

packages = Dict{String,Union{Nothing, VersionNumber}}(
	"Distributions"=>v"0.19.2",
	"Combinatorics"=>v"1.0.0",
	"JSON"=>v"0.21.0",
	"StatsBase"=>v"0.32.2",
	"DataStructures"=>v"0.17.0",
	"Plots"=>v"0.29.5",
	"PyPlot"=>v"2.8.2",
	"Einsum"=>v"0.4.1",
	"Clustering"=>v"0.13.3",
	"SpecialFunctions"=>v"0.8.0"
	)

if "latest" in ARGS
	package_specs = [Pkg.PackageSpec(name=name) for (name, version) in packages]
else
	package_specs = [Pkg.PackageSpec(;name=name, version=version) for (name, version) in packages]
end

Pkg.add(package_specs)
Pkg.build(package_specs)