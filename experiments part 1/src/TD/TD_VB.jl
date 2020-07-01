module TD_VB

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions

export standard_VB

function standard_VB(X::Array{ℜ,3}, R; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    # setting up the dimensions, hyperparameters, priors
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    R1, R2, R3 = typeof(R) == Int ? (R, R, R) : R
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/(R1*R2*R3),R1,R2,R3), fill(a/(I*R1),I,R1), fill(a/(J*R2),J,R2), fill(a/(K*R3),K,R3)
    # declaring the expectations for various random variables
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = reshape(log.(rand(Dirichlet(vec(α_R) .+ 1.0/(R1*R2*R3)))), R1,R2,R3)
    log_θ_IR = reshape([log(θ_ir) for r=1:R1 for θ_ir=rand(Dirichlet(α_IR[:,r] .+ 1.0/I))],I,R1)
    log_θ_JR = reshape([log(θ_jr) for r=1:R2 for θ_jr=rand(Dirichlet(α_JR[:,r] .+ 1.0/J))],J,R2)
    log_θ_KR = reshape([log(θ_kr) for r=1:R3 for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K))],K,R3)
    
    # declaring the marginals of the S tensor
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R1,R2,R3), zeros(ℜ,R1,R2,R3), zeros(ℜ,I,R1), zeros(ℜ,J,R2), zeros(ℜ,K,R3), sum(X)
    q, log_ρ, log_ρ_ijk::ℜ = Array{ℜ}(undef,R1,R2,R3), Array{ℜ}(undef,R1,R2,R3), 0.0
    ELBO::Array{ℜ} = zeros(ℜ,EPOCHS)                     
    
    # This expectation never changes so we compute it first
    log_λ = digamma(S⁺+a) - log(b+1.0)
    for eph=1:EPOCHS
        S_R .= 0.0
        S_IR .= 0.0
        S_JR .= 0.0
        S_KR .= 0.0
                                        
        for k=1:K, j=1:J, i=1:I #order of traversal is important
            # computing the parameters for q(S)
            log_ρ .= log_λ .+ log_θ_R .+ (reshape(log_θ_IR[i,:], R1, 1, 1) .+ 
                                         reshape(log_θ_JR[j,:], 1, R2, 1) .+ 
                                         reshape(log_θ_KR[k,:], 1, 1, R3))
            log_ρ_ijk = logsumexp(log_ρ)
            q .= exp.(log_ρ .- log_ρ_ijk)
            
            # computing E(S)
            s .= X[i,j,k] .* q
            S_R .+= s
            S_IR[i,:] .+= vec(sum(s, dims=[2,3]))
            S_JR[j,:] .+= vec(sum(s, dims=[1,3]))
            S_KR[k,:] .+= vec(sum(s, dims=[1,2]))
            # computing the ELBO
            # nonbookish implementation of this line:   ELBO[eph] += sum(-s .* (log_ρ .- log_ρ_ijk)) - loggamma(X[i,j,k] + 1.0)    
            ELBO[eph] += sum((-X[i,j,k] .* q) .* (log_ρ .- log_ρ_ijk)) - loggamma(X[i,j,k] + 1.0)
        end
        S_R1, S_R2, S_R3 = vec(sum(S_R, dims=[2,3])), vec(sum(S_R, dims=[1,3])), vec(sum(S_R, dims=[1,2]))
        α_R1, α_R2, α_R3 = vec(sum(α_R, dims=[2,3])), vec(sum(α_R, dims=[1,3])), vec(sum(α_R, dims=[1,2]))
        # computing the ELBO
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) + sum(loggamma.(α_R + S_R)) - sum(loggamma.(α_R))# parts that cancelled  + loggamma(a + S⁺) - loggamma(a)   + loggamma(a)  - loggamma(S⁺ + a)
        ELBO[eph] += sum(loggamma.(α_IR + S_IR)) - sum(loggamma.(α_R1 + S_R1)) + sum(loggamma.(α_R1)) - sum(loggamma.(α_IR))
        ELBO[eph] += sum(loggamma.(α_JR + S_JR)) - sum(loggamma.(α_R2 + S_R2)) + sum(loggamma.(α_R2)) - sum(loggamma.(α_JR))
        ELBO[eph] += sum(loggamma.(α_KR + S_KR)) - sum(loggamma.(α_R3 + S_R3)) + sum(loggamma.(α_R3)) - sum(loggamma.(α_KR))
        """Providing all terms here for code review:
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) + loggamma(a + S⁺) - loggamma(a) + sum(loggamma.(α_R + S_R)) - log(S⁺ + a) + loggamma(a) - sum(loggamma.(α_R))
        ELBO[eph] += sum(loggamma.(α_IR + S_IR)) - sum(loggamma.(α_R + S_R)) + sum(loggamma.(α_R)) - sum(loggamma.(α_IR)) # Repeat line for J and K
        ...
        """
        # computing and E(theta)
        log_θ_R .= digamma.(S_R.+α_R) .- digamma(S⁺+a)
        log_θ_IR .= digamma.(S_IR.+α_IR) .- digamma.(S_R1.+α_R1)'
        log_θ_JR .= digamma.(S_JR.+α_JR) .- digamma.(S_R2.+α_R2)'
        log_θ_KR .= digamma.(S_KR.+α_KR) .- digamma.(S_R3.+α_R3)'
    end
    θs = (log_θ_R, log_θ_IR, log_θ_JR, log_θ_KR)
    flat_θs = Tuple(vec(θ) for θ in θs)
    return ELBO, X, log_λ, flat_θs
    end;


end 