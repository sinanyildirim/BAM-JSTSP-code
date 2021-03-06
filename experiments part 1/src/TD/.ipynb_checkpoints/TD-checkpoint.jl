module TD

#include("CP_Exact.jl")
#include("CP_EM.jl")
include("TD_VB.jl")
include("TD_MC.jl")
include("BackwardKernel.jl")

using .TD_VB, .TD_MC, .BackwardKernel #.CP_Exact, .CP_EM, 

#export generate, log_marginal
#export standard_EM, dual_EM 
export standard_VB
export Particle, smc_weight #, particle_mcmc
export EventQueue, sum, length, eltype, iterate

end