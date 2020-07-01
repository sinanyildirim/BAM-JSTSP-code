module TD_Exact

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions, Einsum, Combinatorics
import Base.Iterators: product

export log_marginal

allocations(X::Array{ℜ,N}, H::Vararg{Ƶ}) where {ℜ<:Real, Ƶ<:Integer, N} = Channel(ctype=Array{ℜ,N+length(H)}) do c
    I, R = size(X), prod(H)
    for partition = product(map(Xᵢ -> combinations(1:(Xᵢ+R-1),R-1), X)...)
        push!(c,reshape([sᵢ for (i,pᵢ) ∈ enumerate(partition)
                        for sᵢ = diff([0, pᵢ..., X[i]+R]) .- 1],H...,I...))
    end
end

function log_marginal(X::Array{ℜ,N}, R::Tuple; a=1.0, b=a/nansum(X)) where {ℜ<:Real, Ƶ<:Integer, N}
    I, J, K = size(X)
    R1, R2, R3 = R

    α_R1I, α_R2J, α_R3K = fill(a/(R1*I),R1,I), fill(a/(R2*J),R2,J), fill(a/(R3*K),R3,K)
    α_R1, α_R1R2, α_R1R2R3 = fill(a/(R1),R1), fill(a/(R1*R2),R1,R2), fill(a/(R3*R2*R1),R1,R2,R3)
    S_R1I, S_R2J, S_R3K = Array{Float64}(undef, R1, I), Array{Float64}(undef, R2, J), Array{Float64}(undef, R3, K)
    S_R1, S_R1R2, S_R1R2R3 = Array{Float64}(undef, R1), Array{Float64}(undef, R1, R2), Array{Float64}(undef, R1, R2, R3)
    S₊ = sum(X)
    
    log_PX = -Inf
    log_C = a*log(b) - (a+S₊)*log(b + 1) + lgamma(a + S₊) - lgamma(a)

    for S ∈ allocations(X,R...)
        log_PS = log_C - sum(lgamma, S .+ 1)

        S_R1I, S_R2J, S_R3K  = sum(S, (2,3,5,6)), sum(S, (1,3,4,6)), sum(S, (1,2,4,5))
        S_R1, S_R1R2, S_R1R2R3 = sum(S, (2,3,4,5,6)), sum(S, (3,4,5,6)), sum(S, (4,5, 6)) 

        log_PS +=
        sum(loggamma.(S_R1I .+ α_R1I)) - sum(loggamma.(sum(S_R1I .+ α_R1I, 2))) + sum(loggamma.(sum(α_R1I, 2))) - sum(loggamma.(α_R1I)) +
        sum(loggamma.(S_R2J .+ α_R2J)) - sum(loggamma.(sum(S_R2J .+ α_R2J, 2))) + sum(loggamma.(sum(α_R2J, 2))) - sum(loggamma.(α_R2J)) +
        sum(loggamma.(S_R3K .+ α_R3K)) - sum(loggamma.(sum(S_R3K .+ α_R3K, 2))) + sum(loggamma.(sum(α_R3K, 2))) - sum(loggamma.(α_R3K)) +
        sum(loggamma.(S_R1     .+ α_R1))     - loggamma.(S₊ + a)                            + loggamma.(a)                     - sum(loggamma.(α_R1)) +
        sum(loggamma.(S_R1R2   .+ α_R1R2))   - sum(loggamma.(sum(S_R1R2   .+ α_R1R2, 2)))   + sum(loggamma.(sum(α_R1R2, 2)))   - sum(loggamma.(α_R1R2)) +
        sum(loggamma.(S_R1R2R3 .+ α_R1R2R3)) - sum(loggamma.(sum(S_R1R2R3 .+ α_R1R2R3, 3))) + sum(loggamma.(sum(α_R1R2R3, 3))) - sum(loggamma.(α_R1R2R3))

        log_PX = logsumexp([log_PX,log_PS])
    end
    return log_PX
end

end