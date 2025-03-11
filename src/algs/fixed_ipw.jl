using Distributions
using ..AdaptiveATE: CONTROL, TREATMENT

"""
    FixedIPWAllocation <: IPWAlgorithm

Fixed allocation algorithm that maintains constant sampling probabilities, using IPW estimation.

# Fields
- `t::Int`: Current time step
- `π̂::Float64`: Fixed probability of selecting arm 2
"""
mutable struct FixedIPWAllocation <: IPWAlgorithm
    t::Int
    π̂::Float64
end

function FixedIPWAllocation(π::Float64=0.5)
    if π < 0.0 || π > 1.0
        error("Fixed allocation probability must be between 0 and 1")
    end
    return FixedIPWAllocation(0, π)
end

"""
    IPWNeymanAllocation(m²::Vector{Float64})

Construct a fixed IPW allocation algorithm using Neyman allocation based on second moments.

# Arguments
- `m²::Vector{Float64}`: Vector of second moments for each arm
"""
function IPWNeymanAllocation(m²::Vector{Float64})
    if length(m²) != 2
        error("Second moments vector must be of length 2")
    end
    return FixedIPWAllocation(neyman_allocation(sqrt.(m²)))
end

function update!(self::FixedIPWAllocation, A::Int, R::Real)
    # Do nothing except increment time
    self.t += 1
end

function reset!(self::FixedIPWAllocation)
    self.t = 0
end

function decide(self::FixedIPWAllocation, context=nothing)
    A = rand(Bernoulli(self.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? self.π̂ : 1 - self.π̂
    return A, πA, 0.0, 0.0  # Return action, probability, reward_estimate, direct_estimate
end 