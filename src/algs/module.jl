module Algorithms

using ..AdaptiveATE: CONTROL, TREATMENT
using ..Utils: streaming_mean, clip, poly_decay, neyman_allocation  # Add this line to import utility functions

using Distributions  # Add this since clipSMT.jl needs it
using StaticArrays

export OPEAlgorithm, IPWAlgorithm, AIPWAlgorithm # Abstract types
export ClipSMT, ClipOGD, EtC, FixedIPWAllocation, FTRL # IPW algorithms
export FixedAllocationFixedReward, NeymanAllocationTrueReward, FixedAllocationEstimateReward, NeymanAllocationEstimateReward # Fixed AIPW algorithms
export ClipSDT, OPT # Other AIPW algorithms
export ClipSMTSIMD
export decide, update!, reset! # Methods
export ALG_NAMES, default_alg_constructor 

### Algorithm Factory

ALG_NAMES = [
    "ClipSMT", 
    "ClipOGD", 
    "EtC", 
    "ClipSDT", 
    "OPTrack",
    "IPW Neyman Allocation",  
    "NA Est",
    "NeymanAllocationTrueReward",
    "FTRL"
    ]
function default_alg_constructor(alg_name::String, num_rounds::Int, m²::Vector{Float64}, means::Vector{Float64}=zeros(2))
    if alg_name == "ClipSMT"
        return ClipSMT()
    elseif alg_name == "ClipOGD"
        return ClipOGD(num_rounds)
    elseif alg_name == "Balanced IPW Allocation"
        return FixedIPWAllocation(0.5)
    elseif alg_name == "IPW Neyman Allocation"
        return IPWNeymanAllocation(m²)
    elseif alg_name == "EtC"
        return EtC(num_rounds, 0.5)
    elseif alg_name == "ClipSDT"
        return ClipSDT(1.0 / 3.0)
    elseif alg_name == "OPTrack"
        return OPT(0.1)  # Default confidence parameter of 0.1
    elseif alg_name == "NA Est"
        σ² = m² .- means.^2  # Convert second moments to variances
        return NeymanAllocationEstimateReward(σ²)
    elseif alg_name == "NeymanAllocationTrueReward"
        return NeymanAllocationTrueReward(m² .- means.^2, means)
    elseif alg_name == "FTRL"
        return FTRL()
    else
        throw(ArgumentError("Unknown algorithm name: $alg_name\nValid names: $ALG_NAMES"))
    end
end 

### Generic OPE Algorithm Hierarchy
abstract type OPEAlgorithm end
# IPW algorithms use raw second moments for Neyman allocation
abstract type IPWAlgorithm <: OPEAlgorithm end
# AIPW algorithms use standard deviations for Neyman allocation
abstract type AIPWAlgorithm <: OPEAlgorithm end

"""
    decide(alg::OPEAlgorithm, context=nothing)

Make a decision and return action, probability of that action, reward estimate, and direct estimate.

# Arguments
- `alg::OPEAlgorithm`: The algorithm instance
- `context`: Optional context for the decision (default: nothing)

# Returns
- `Tuple{Int, Float64, Float64, Float64}`: (action, probability of that action, reward_estimate, direct_estimate)
"""
function decide(alg::IPWAlgorithm, context=nothing)
    A = rand(Bernoulli(alg.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? alg.π̂ : 1 - alg.π̂
    return A, πA, 0.0, 0.0
end

function update!(alg::OPEAlgorithm, arm::Int, reward::Float64)
    error("update! is not implemented for $(typeof(alg))")
end

function reset!(alg::OPEAlgorithm)
    error("reset! is not implemented for $(typeof(alg))")
end

### Concrete Bandit Algorithms
include("clipSMT.jl")
include("clipOGD.jl")
include("fixed_ipw.jl")
include("fixed_aipw.jl")  # New file for fixed AIPW algorithms
include("EtC.jl")
include("clipSDT.jl")
include("OPT.jl")
include("ftrl.jl")

end # module Algorithms