module TD_MC

include("../Misc.jl")
include("BackwardKernel.jl")
using .Misc, .BackwardKernel

using Distributions, SpecialFunctions

export Particle, smc_weight, particle_mcmc

mutable struct Particle{ℜ <: Real}
    S_R::Array{ℜ,3}
    S_IR::Array{ℜ,2}
    S_JR::Array{ℜ,2}
    S_KR::Array{ℜ,2}
    function Particle(I::Int,J::Int, K::Int, R1::Int, R2::Int, R3::Int; γ::ℜ=0.1) where {ℜ<:Real}
        a::ℜ = I*J*K*γ
        return new{ℜ}(fill(a/(R1*R2*R3),R1,R2,R3),  fill(a/(R1*I),I,R1), fill(a/(R2*J),J,R2), fill(a/(R3*K),K,R3))
    end
    function Particle(S_R::Array{ℜ,3}, S_IR::Array{ℜ,2}, S_JR::Array{ℜ,2}, S_KR::Array{ℜ,2}) where {ℜ<:Real}
        return new{ℜ}(S_R, S_IR, S_JR, S_KR)
    end
end

function smc_weight(X::Array{ℜ,3}, R, N::Ƶ=1; μ::ℜ=3.5, γ::ℜ=0.1, adaptive::Bool=true, resampling_freq::Int=1, resampling_scheme="systematic", observation_order=NaN) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    R1, R2, R3 = typeof(R) == Int ? (R, R, R) : R
    T::Ƶ = Ƶ(sum(X))
    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)
    if !adaptive & (resampling_freq > 1)
        ESS_min = fill(0.,T)
        ESS_min[resampling_freq:resampling_freq:end] .= N+1.0
    end

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    
    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0) - sum(loggamma.(X .+ 1.0))
    
    P = [Particle(I,J,K,R1,R2,R3; γ=γ) for n=1:N]
    P_temp = [Particle(I,J,K,R1,R2,R3; γ=γ) for n=1:N]
    log_wᵗ::Array{ℜ}, Wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(1.0/N,N), fill(-log(N),N), zeros(ℜ,N)

    log_νᵗ::ℜ, log_q_r::Array{ℜ,3}, q_r::Array{ℜ,3}= 0.0, zeros(ℜ,R1,R2,R3), zeros(ℜ,R1,R2,R3)
    rᵗ::Ƶ, iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, uᵗ::ℜ  = 0, 0, 0, 0, 0.0

    backward_sampler = typeof(observation_order) <: Array ? observation_order : EventQueue(X)
    for (t,(iᵗ, jᵗ, kᵗ)) in enumerate(backward_sampler)
        for (n,p) in enumerate(P)
            S_R1, S_R2, S_R3 = vec(sum(p.S_R, dims=[2,3])), vec(sum(p.S_R, dims=[1,3])), vec(sum(p.S_R, dims=[1,2]))
            log_q_r .= log.(p.S_R).+ (reshape((log.(p.S_IR[iᵗ,:]) .- log.(S_R1)), R1, 1, 1) .+ 
                                     reshape((log.(p.S_JR[jᵗ,:]) .- log.(S_R2)), 1, R2, 1) .+ 
                                     reshape((log.(p.S_KR[kᵗ,:]) .- log.(S_R3)), 1, 1, R3))
            log_νᵗ = logsumexp(log_q_r)
            log_q_r .-= log_νᵗ
            q_r .= exp.(log_q_r)

            # Check this vectorization
            rᵗ = rand(Categorical(vec(q_r)))
            r1ᵗ, r2ᵗ, r3ᵗ = Tuple(CartesianIndices(q_r)[rᵗ])
            p.S_R[r1ᵗ, r2ᵗ, r3ᵗ] += 1.0 
            p.S_IR[iᵗ,r1ᵗ] += 1.0 
            p.S_JR[jᵗ,r2ᵗ] += 1.0 
            p.S_KR[kᵗ,r3ᵗ] += 1.0 

            log_wᵗ[n] += log_νᵗ
        end
        
        log_Zᵗ = logsumexp(log_wᵗ)
        log_Wᵗ .= log_wᵗ .- log_Zᵗ
        log_Zᵗ -= log(N)
        Wᵗ .= exp.(log_Wᵗ)

        ESS[t] = 1.0/sum(Wᵗ .* Wᵗ)
        
        if ESS[t] < ESS_min[t]
            for n=1:N
                P_temp[n].S_R .= P[n].S_R
                P_temp[n].S_IR .= P[n].S_IR
                P_temp[n].S_JR .= P[n].S_JR
                P_temp[n].S_KR .= P[n].S_KR
            end
            
            # refactor this function out
            if resampling_scheme == "systematic"
                cum_Wᵗ .= cumsum(Wᵗ)
                uᵗ = rand()/N
                p_id = 1

                for n=1:N # systematic resampling
                    while cum_Wᵗ[p_id] < uᵗ
                        p_id += 1
                    end
                    P[n].S_R .= P_temp[p_id].S_R
                    P[n].S_IR .= P_temp[p_id].S_IR
                    P[n].S_JR .= P_temp[p_id].S_JR
                    P[n].S_KR .= P_temp[p_id].S_KR
                    uᵗ += 1.0/N
                end
            elseif resampling_scheme == "multinomial"
                for n=1:N
                    p_id = rand(Categorical(Wᵗ))
                    P[n].S_R .= P_temp[p_id].S_R
                    P[n].S_IR .= P_temp[p_id].S_IR
                    P[n].S_JR .= P_temp[p_id].S_JR
                    P[n].S_KR .= P_temp[p_id].S_KR
                end
            end
            log_wᵗ .= log_Zᵗ
        end
    end
    p_id = rand(Categorical(Wᵗ))
    P[p_id].S_R .-= a/(R1*R2*R3)
    P[p_id].S_IR .-= a/(R1*I)
    P[p_id].S_JR .-= a/(R2*J)
    P[p_id].S_KR .-= a/(R3*K)
    return log_Zᵗ, ESS, P[p_id]
end

end