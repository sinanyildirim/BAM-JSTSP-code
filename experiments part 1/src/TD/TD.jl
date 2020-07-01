module TD

include("TD_Exact.jl")
include("TD_VB.jl")
include("TD_MC.jl")
include("BackwardKernel.jl")

using .TD_VB, .TD_MC, .BackwardKernel, .TD_Exact
export generate, log_marginal
export standard_VB
export Particle, smc_weight #, particle_mcmc
export EventQueue, sum, length, eltype, iterate

end
