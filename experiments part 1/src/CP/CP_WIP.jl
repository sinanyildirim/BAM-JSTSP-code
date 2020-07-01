"""
Auxuilary, work-in-progress functions for CP.
"""

"""
A refactored version of CP SMC
"""

function get_dimensions(X, R)
    R = (typeof(R) <: Tuple) && (length(R) == 1) ? R[1] : R;
    @assert typeof(R) == Int # HACK: If the R is provided as tuple of length 1
    I, J, K = size(X)
    return I, J, K, R
end

function get_ESS_min(adaptive, N, T, resampling_freq)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)
    if !adaptive & (resampling_freq > 1)
        ESS_min = fill(0.,T)
        ESS_min[resampling_freq:resampling_freq:end] .= N+1.0
    end
    return ESS_min
end

function increment_marginals!(p::Particle, iᵗ::Int, jᵗ::Int, kᵗ::Int, rᵗ::Int)
    p.S_R[rᵗ] += 1.0
    p.S_IR[iᵗ,rᵗ] += 1.0
    p.S_JR[jᵗ,rᵗ] += 1.0
    p.S_KR[kᵗ,rᵗ] += 1.0
end

function get_q_r(p::Particle, iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, log_Zᵗ::ℜ)::Array{ℜ} where {ℜ<:Real, Ƶ<:Int}
    log_q_r::Array{ℜ} .= log.(p.S_IR[iᵗ,:]) .+ log.(p.S_JR[jᵗ,:]) .+ log.(p.S_KR[kᵗ,:]) .- 2.0.*log.(p.S_R)
    log_νᵗ::ℜ = logsumexp(log_q_r)
    log_q_r .-= log_νᵗ
    q_r::ℜ .= exp.(log_q_r)
    return q_r
end

function update_Z_get_W(log_wᵗ::ℜ, N::Ƶ) where {ℜ<:Real, Ƶ<:Int}
    log_Zᵗ::ℜ = logsumexp(log_wᵗ)
    log_Wᵗ::ℜ .= log_wᵗ .- log_Zᵗ
    log_Zᵗ -= log(N)
    Wᵗ::ℜ .= exp.(log_Wᵗ)
    return log_Zᵗ, Wᵗ

function resampling!(P, P_temp, Wᵗ, resampling_scheme)
    N = length(Wᵗ)
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
end

function sample_and_prepare_particle(P::Array{Particle}, Wᵗ::Array{ℜ}, a::ℜ, I::Ƶ, J::Ƶ, K::Ƶ)  where {ℜ<:Real, Ƶ<:Int}
    p_id = rand(Categorical(Wᵗ))
    P[p_id].S_R .-= a/R
    P[p_id].S_IR .-= a/(R*I)
    P[p_id].S_JR .-= a/(R*J)
    P[p_id].S_KR .-= a/(R*K)
    return P[p_id]
end

function smc_weight(X::Array{ℜ,3}, R, N::Ƶ=1; μ::ℜ=nanmean(X), γ::ℜ=0.1, adaptive::Bool=true, resampling_freq::Int=1, resampling_scheme="systematic", observation_order=NaN) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ, R = get_dimensions(X, R)
    T::Ƶ = Ƶ(sum(X))
    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = get_ESS_min(adaptive, N, T, resampling_freq)
    a::ℜ, b::ℜ, P, P_temp = I*J*K*γ, γ/μ, [Particle(I,J,K,R; γ=γ) for n=1:N], [Particle(I,J,K,R; γ=γ) for n=1:N]

    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0) - sum(loggamma.(X .+ 1.0))
    log_wᵗ::Array{ℜ}, Wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(1.0/N,N), fill(-log(N),N), zeros(ℜ,N)
    q_r::Array{ℜ}, rᵗ::Ƶ, iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, uᵗ::ℜ = zeros(ℜ,R), 0, 0, 0, 0, 0.0

    backward_sampler = typeof(observation_order) <: Array ? observation_order : EventQueue(X)
    for (t,(iᵗ, jᵗ, kᵗ)) in enumerate(backward_sampler)
        for (n,p) in enumerate(P)
            q_r = get_q_r(p, iᵗ, jᵗ, kᵗ, log_Zᵗ)
            rᵗ = rand(Categorical(q_r))
            increment_marginals!(p, iᵗ, jᵗ, kᵗ, rᵗ)
            log_wᵗ[n] += log_νᵗ
        end
        log_Zᵗ, Wᵗ = update_Z_get_W(log_wᵗ, N)

        ESS[t] = 1.0/sum(Wᵗ .* Wᵗ)
        if ESS[t] < ESS_min[t]
            resampling!(P, P_temp, Wᵗ, resampling_scheme)
            log_wᵗ .= log_Zᵗ
        end
    end
    p = sample_and_prepare_particle(P, Wᵗ, a, I, J, K)
    return log_Zᵗ, ESS, p
end

"""
An SMC version where resampling is conducted in the middle of the algorithm
"""

mutable struct Particle_resampling{ℜ <: Real}
    S_R::Array{ℜ,1}
    S_IR::Array{ℜ,2}
    S_JR::Array{ℜ,2}
    S_KR::Array{ℜ,2}
    q_r::Array{ℜ,1}
    function Particle_resampling(I::Int,J::Int, K::Int, R::Int; γ::ℜ=0.1) where {ℜ<:Real}
        a::ℜ = I*J*K*γ
        return new{ℜ}(fill(a/R,R), fill(a/(R*I),I,R), fill(a/(R*J),J,R), fill(a/(R*K),K,R), fill(0.0, R))
    end
    function Particle_resampling(S_R::Array{ℜ,1}, S_IR::Array{ℜ,2}, S_JR::Array{ℜ,2}, S_KR::Array{ℜ,2}) where {ℜ<:Real}
        return new{ℜ}(S_R, S_IR, S_JR, S_KR, fill(0.0, length(S_R)))
    end
end


function smc_weight_resampling(X::Array{ℜ,3}, R::Ƶ, N::Ƶ=1; μ::ℜ=nanmean(X), γ::ℜ=0.1, adaptive::Bool=true, resampling_freq::Int=1, resampling_scheme="systematic") where {ℜ<:Real, Ƶ<:Int}
    """
    Resampling conducted in the middle instead of at the end
    """
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    T::Ƶ = Ƶ(sum(X))
    ESS::Array{ℜ} = zeros(ℜ,T)
    ESS_min::Array{ℜ} = adaptive ? fill(N/2.0,T) : fill(N+1.0,T)
    if !adaptive & (resampling_freq > 1)
        ESS_min = fill(0.,T)
        ESS_min[resampling_freq:resampling_freq:end] .= N+1.0
    end

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    log_Zᵗ::ℜ = a*log(b) - (a+T)*log(b + 1.0) - sum(loggamma.(X .+ 1.0))
    P = [Particle_resampling(I,J,K,R; γ=γ) for n=1:N]
    P_temp = [Particle_resampling(I,J,K,R; γ=γ) for n=1:N]
    log_wᵗ::Array{ℜ}, Wᵗ::Array{ℜ}, log_Wᵗ::Array{ℜ}, cum_Wᵗ::Array{ℜ} = fill(log_Zᵗ,N), fill(1.0/N,N), fill(-log(N),N), zeros(ℜ,N)

    log_νᵗ::ℜ, log_q_r::Array{ℜ} = 0.0, zeros(ℜ,R) # q_r::Array{ℜ}= zeros(ℜ,R),  --q_r now at each particle
    rᵗ::Ƶ, iᵗ::Ƶ, jᵗ::Ƶ, kᵗ::Ƶ, uᵗ::ℜ  = 0, 0, 0, 0, 0.0

    for (t,(iᵗ, jᵗ, kᵗ)) in enumerate(EventQueue(X))
        for (n,p) in enumerate(P)

            log_q_r .= log.(p.S_IR[iᵗ,:]) .+ log.(p.S_JR[jᵗ,:]) .+ log.(p.S_KR[kᵗ,:]) .- 2.0.*log.(p.S_R)
            log_νᵗ = logsumexp(log_q_r)
            log_q_r .-= log_νᵗ
            p.q_r .= exp.(log_q_r)

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
                P_temp[n].q_r .= P[n].q_r
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
                    P[n].q_r .= P_temp[p_id].q_r
                    uᵗ += 1.0/N
                end
            elseif resampling_scheme == "multinomial"
                for n=1:N
                    p_id = rand(Categorical(Wᵗ))
                    P[n].S_R .= P_temp[p_id].S_R
                    P[n].S_IR .= P_temp[p_id].S_IR
                    P[n].S_JR .= P_temp[p_id].S_JR
                    P[n].S_KR .= P_temp[p_id].S_KR
                    P[n].q_r .= P_temp[n].q_r
                end
            end
            log_wᵗ .= log_Zᵗ
        end
        for (n,p) in enumerate(P)
            rᵗ = rand(Categorical(p.q_r))
            p.S_R[rᵗ] += 1.0
            p.S_IR[iᵗ,rᵗ] += 1.0
            p.S_JR[jᵗ,rᵗ] += 1.0
            p.S_KR[kᵗ,rᵗ] += 1.0
        end
    end
    p_id = rand(Categorical(Wᵗ))
    P[p_id].S_R .-= a/R
    P[p_id].S_IR .-= a/(R*I)
    P[p_id].S_JR .-= a/(R*J)
    P[p_id].S_KR .-= a/(R*K)
    return log_Zᵗ, ESS, P[p_id]
end
