module CP_VB

include("../Misc.jl")
using .Misc

using Distributions, SpecialFunctions

export standard_VB, online_VB

function standard_VB(X::Array{ℜ,3}, R::Ƶ ; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    # setting up the dimensions, hyperparameters, priors
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    # declaring the expectations for various random variables
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = log.(rand(Dirichlet(α_R .+ 1.0/R)))
    log_θ_IR = reshape([log(θ_ir) for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r] .+ 1.0/I))],I,R)
    log_θ_JR = reshape([log(θ_jr) for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r] .+ 1.0/J))],J,R)
    log_θ_KR = reshape([log(θ_kr) for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K))],K,R)
    
    # declaring the marginals of the S tensor
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), zeros(ℜ,J,R), zeros(ℜ,K,R), sum(X)
    q, log_ρ, log_ρ_ijk::ℜ = Array{ℜ}(undef,R), Array{ℜ}(undef,R), 0.0
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
            log_ρ .= log_λ .+ log_θ_R .+ log_θ_IR[i,:] .+ log_θ_JR[j,:] .+ log_θ_KR[k,:]
            log_ρ_ijk = logsumexp(log_ρ)
            q .= exp.(log_ρ .- log_ρ_ijk)
            
            # computing E(S)
            s .= X[i,j,k] .* q
            S_R .+= s
            S_IR[i,:] .+= s
            S_JR[j,:] .+= s
            S_KR[k,:] .+= s
            # computing the ELBO
            ELBO[eph] += sum(-s .* (log_ρ .- log_ρ_ijk)) - loggamma(X[i,j,k] + 1.0)    
        end
        # computing the ELBO (commenting out the terms that cancel for ease of code review)
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) + 2*sum(loggamma.(α_R))  - 2*sum(loggamma.(α_R + S_R))
        ELBO[eph] += sum(loggamma.(α_IR + S_IR)) - sum(loggamma.(α_IR)) + sum(loggamma.(α_JR + S_JR)) - sum(loggamma.(α_JR)) + sum(loggamma.(α_KR + S_KR)) - sum(loggamma.(α_KR)) 
        """Providing all terms here for code review:
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) + loggamma(a + S⁺) - loggamma(a) + sum(loggamma.(α_R + S_R)) - loggamma(S⁺ + a) + loggamma(a) - sum(loggamma.(α_R))
        ELBO[eph] += sum(loggamma.(α_IR + S_IR)) - sum(loggamma.(α_R + S_R)) + sum(loggamma.(α_R)) - sum(loggamma.(α_IR)) # Repeat line for J and K
        """
        # computing and E(theta)
        log_θ_R .= digamma.(S_R.+α_R) .- digamma(S⁺+a)
        log_θ_IR .= digamma.(S_IR.+α_IR) .- digamma.(S_R.+α_R)'
        log_θ_JR .= digamma.(S_JR.+α_JR) .- digamma.(S_R.+α_R)'
        log_θ_KR .= digamma.(S_KR.+α_KR) .- digamma.(S_R.+α_R)'

    end
    θs = (log_θ_R, log_θ_IR, log_θ_JR, log_θ_KR)
    flat_θs = Tuple(vec(θ) for θ in θs)
    return ELBO, X, log_λ, flat_θs
end

function standard_VB_missing(X::Array{ℜ,3}, R::Ƶ ; μ::ℜ=3.5, γ::ℜ=0.1, EPOCHS::Ƶ=1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    """
    This is the previous implementation of VB, that includes missing data and is .20 faster than the implementation above. 
    """
    # setting up the dimensions, hyperparameters, priors
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)
    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    # declaring the expectations for various random variables
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = log.(rand(Dirichlet(α_R .+ 1.0/R)))
    log_θ_IR = reshape([log(θ_ir) for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r] .+ 1.0/I))],I,R)
    log_θ_JR = reshape([log(θ_jr) for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r] .+ 1.0/J))],J,R)
    log_θ_KR = reshape([log(θ_kr) for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K))],K,R)
    
    # declaring the marginals of the S tensor
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), zeros(ℜ,J,R), zeros(ℜ,K,R), sum(X)
    q, log_ρ, log_ρ_ijk::ℜ = Array{ℜ}(undef,R), Array{ℜ}(undef,R), 0.0
    ELBO::Array{ℜ} = zeros(ℜ,EPOCHS)                     
    X_full = similar(X)

    for eph=1:EPOCHS
        S_R .= 0.0
        S_IR .= 0.0
        S_JR .= 0.0
        S_KR .= 0.0
        S⁺ = 0.0
                                        
        for k=1:K, j=1:J, i=1:I #order of traversal is important
            # computing the parameters for q(S)
            log_ρ .= log_λ .+ log_θ_R .+ log_θ_IR[i,:] .+ log_θ_JR[j,:] .+ log_θ_KR[k,:]
            log_ρ_ijk = logsumexp(log_ρ)
            q .= exp.(log_ρ .- log_ρ_ijk)
            
            # computing E(S)
            X_full[i,j,k] = isnan(X[i,j,k]) ? exp(log_ρ_ijk) : X[i,j,k]
            s .= X_full[i,j,k] .* q
            S_R .+= s
            S_IR[i,:] .+= s
            S_JR[j,:] .+= s
            S_KR[k,:] .+= s
            S⁺ += X_full[i,j,k]
            # computing the ELBO
            ELBO[eph] += isnan(X[i,j,k]) ? X_full[i,j,k] : X[i,j,k]*log_ρ_ijk - loggamma(X[i,j,k] + 1.0)
        end
        # computing the ELBO
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+1.0) - sum(loggamma.(α_IR)) + sum(loggamma.(α_IR .+ S_IR))
        ELBO[eph] += sum(loggamma.(α_JR .+ S_JR)) - sum(loggamma.(α_JR)) + sum(loggamma.(α_KR .+ S_KR)) - sum(loggamma.(α_KR))
        ELBO[eph] += 2.0*sum(loggamma.(α_R)) - 2.0*sum(loggamma.(α_R .+ S_R))
        ELBO[eph] -= S⁺ * log_λ + sum(S_R .* log_θ_R) + sum(S_IR .* log_θ_IR) + sum(S_JR .* log_θ_JR) + sum(S_KR .* log_θ_KR)
        
        # computing E(λ) and E(theta)
        log_λ = digamma(S⁺+a) - log(b+1.0)
        log_θ_R .= digamma.(S_R.+α_R) .- digamma(S⁺+a)
        log_θ_IR .= digamma.(S_IR.+α_IR) .- digamma.(S_R.+α_R)'
        log_θ_JR .= digamma.(S_JR.+α_JR) .- digamma.(S_R.+α_R)'
        log_θ_KR .= digamma.(S_KR.+α_KR) .- digamma.(S_R.+α_R)'

    end
    θs = (log_θ_R, log_θ_IR, log_θ_JR, log_θ_KR)
    flat_θs = Tuple(vec(θ) for θ in θs)
    return ELBO, X_full, log_λ, flat_θs
end

function online_VB(X::Array{ℜ,3}, R::Ƶ, Ω::AbstractArray{ℜ,1}; μ::ℜ=3.5, γ::ℜ=0.1, ϵ::ℜ=1e-16) where {ℜ<:Real, Ƶ<:Int}
    I::Ƶ, J::Ƶ, K::Ƶ = size(X)

    a::ℜ, b::ℜ = I*J*K*γ, γ/μ
    α_R, α_IR, α_JR, α_KR = fill(a/R,R), fill(a/(I*R),I,R), fill(a/(J*R),J,R), fill(a/(K*R),K,R)
    
    log_λ = log(rand(Gamma(a,1.0/b)))
    log_θ_R = log.(rand(Dirichlet(α_R .+ 1.0/R)))
    log_θ_IR = reshape([log(θ_ir) for r=1:R for θ_ir=rand(Dirichlet(α_IR[:,r] .+ 1.0/I))],I,R)
    log_θ_JR = reshape([log(θ_jr) for r=1:R for θ_jr=rand(Dirichlet(α_JR[:,r] .+ 1.0/J))],J,R)
    log_θ_KR = reshape([log(θ_kr) for r=1:R for θ_kr=rand(Dirichlet(α_KR[:,r] .+ 1.0/K))],K,R)
    
    s, S_R, S_IR, S_JR, S_KR, S⁺ = zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,I,R), zeros(ℜ,J,R), zeros(ℜ,K,R), sum(X)
    q, log_ρ, log_ρ_ijk::ℜ = Array{ℜ}(undef,R), Array{ℜ}(undef,R), 0.0

    ELBO::Array{ℜ} = zeros(ℜ,length(Ω))                     
    X_full = similar(X)

    for (eph,ω) in enumerate(Ω)
        S_R .= 0.0
        S_IR .= 0.0
        S_JR .= 0.0
        S_KR .= 0.0
        S⁺ = 0.0
                                        
        for k=1:K, j=1:J, i=1:I #order of traversal is important
            log_ρ .= log_λ .+ log_θ_R .+ log_θ_IR[i,:] .+ log_θ_JR[j,:] .+ log_θ_KR[k,:]
            log_ρ_ijk = logsumexp(log_ρ)
            q .= exp.(log_ρ .- log_ρ_ijk)
                                            
            X_full[i,j,k] = isnan(X[i,j,k]) ? ω*exp(log_ρ_ijk) : rand(Binomial(Ƶ(X[i,j,k]),ω)) 
            s .= X_full[i,j,k] .* q

            S_R .+= s
            S_IR[i,:] .+= s
            S_JR[j,:] .+= s
            S_KR[k,:] .+= s
            S⁺ += X_full[i,j,k]

            ELBO[eph] += isnan(X[i,j,k]) ? X_full[i,j,k] : X[i,j,k]*log_ρ_ijk - loggamma(X[i,j,k] + 1.0)

        end
                                        
        ELBO[eph] += a*log(b) - (S⁺+a)*log(b+ω) - sum(loggamma.(α_IR)) + sum(loggamma.(α_IR .+ S_IR))
        ELBO[eph] += sum(loggamma.(α_JR .+ S_JR)) - sum(loggamma.(α_JR)) + sum(loggamma.(α_KR .+ S_KR)) - sum(loggamma.(α_KR))
        ELBO[eph] += 2.0*sum(loggamma.(α_R)) - 2.0*sum(loggamma.(α_R .+ S_R))
        ELBO[eph] -= S⁺ * log_λ + sum(S_R .* log_θ_R) + sum(S_IR .* log_θ_IR) + sum(S_JR .* log_θ_JR) + sum(S_KR .* log_θ_KR)

        log_λ = digamma(S⁺ + a) - log(b + ω) ## should be checked
        log_θ_R .= digamma.(S_R.+α_R) .- digamma(S⁺+a)
        log_θ_IR .= digamma.(S_IR.+α_IR) .- digamma.(S_R.+α_R)'
        log_θ_JR .= digamma.(S_JR.+α_JR) .- digamma.(S_R.+α_R)'
        log_θ_KR .= digamma.(S_KR.+α_KR) .- digamma.(S_R.+α_R)'

    end
    θs = (log_θ_R, log_θ_IR, log_θ_JR, log_θ_KR)
    flat_θs = Tuple(vec(θ) for θ in θs)
    return ELBO, X_full, log_λ, flat_θs
end

end 