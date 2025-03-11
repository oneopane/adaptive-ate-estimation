using StaticArrays
using Distributions
using ..AdaptiveATE: CONTROL, TREATMENT
using ..Utils: neyman_allocation

"""
    FixedAllocationEstimateReward <: AIPWAlgorithm

Fixed allocation algorithm that uses pre-specified allocation probability but estimates rewards online.

# Fields
- `t::Float64`: Current time step
- `π̂::Float64`: Fixed probability of selecting arm 2
- `μ::MVector{2, Float64}`: Empirical means for each arm
- `N::MVector{2, Float64}`: Number of pulls per arm
"""
mutable struct FixedAllocationEstimateReward <: AIPWAlgorithm
    t::Float64
    π̂::Float64
    μ::MVector{2, Float64}
    N::MVector{2, Float64}
end

function FixedAllocationEstimateReward(π::Float64)
    if π < 0.0 || π > 1.0
        error("Fixed allocation probability must be between 0 and 1")
    end
    return FixedAllocationEstimateReward(0.0, π, @MVector[0.0, 0.0], @MVector[0.0, 0.0])
end


"""
    FixedAllocationAndReward <: AIPWAlgorithm

Fixed allocation algorithm that uses pre-specified allocation probability and reward function.

# Fields
- `t::Int`: Current time step
- `π̂::Float64`: Fixed probability of selecting arm 2
- `μ::Vector{Float64}`: Fixed reward estimates for each arm
"""
mutable struct FixedAllocationFixedReward <: AIPWAlgorithm
    t::Int
    π̂::Float64
    μ::Vector{Float64}

end

function FixedAllocationFixedReward(π::Float64, μ::Vector{Float64})
    if π < 0.0 || π > 1.0
        error("Fixed allocation probability must be between 0 and 1")
    end
    if length(μ) != 2
        error("Fixed reward estimates must be of length 2")
    end
    return FixedAllocationFixedReward(0, π, μ)
end

function NeymanAllocationEstimateReward(σ²::Vector{Float64})
    if length(σ²) != 2
        error("Variance vector must be of length 2")
    end
    return FixedAllocationEstimateReward(neyman_allocation(sqrt.(σ²)))
end

function NeymanAllocationTrueReward(σ²::Vector{Float64}, μ::Vector{Float64})
    if length(σ²) != 2 || length(μ) != 2
        error("Variance and mean vectors must be of length 2")
    end
    return FixedAllocationFixedReward(neyman_allocation(sqrt.(σ²)), μ)
end



function update!(self::FixedAllocationEstimateReward, A::Int, R::Real)
    @inbounds begin
        # Update formulas for maintaining running mean
        self.μ[A] = (self.N[A] * self.μ[A] + R) / (self.N[A] + 1.0)
        self.N[A] += 1.0
        self.t += 1.0
    end
end

function reset!(self::FixedAllocationEstimateReward)
    @inbounds begin
        self.t = 0.0
        fill!(self.μ, 0.0)
        fill!(self.N, 0.0)
        # Note: we don't reset π̂ since it's fixed
    end
end

function decide(self::FixedAllocationEstimateReward, context=nothing)
    A = rand(Bernoulli(self.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? self.π̂ : 1 - self.π̂
    
    # Get reward estimate as empirical mean of selected arm
    reward_estimate = self.μ[A]
    
    # Get direct estimate as difference in empirical means
    direct_estimate = self.μ[TREATMENT] - self.μ[CONTROL]
    
    return A, πA, reward_estimate, direct_estimate
end





function update!(self::FixedAllocationFixedReward, A::Int, R::Real)
    # Do nothing except increment time since both allocation and rewards are fixed
    self.t += 1
end

function reset!(self::FixedAllocationFixedReward)
    self.t = 0
    # Note: we don't reset π̂ or μ since they're fixed
end

function decide(self::FixedAllocationFixedReward, context=nothing)
    A = rand(Bernoulli(self.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? self.π̂ : 1 - self.π̂
    
    # Return fixed reward estimates
    reward_estimate = self.μ[A]
    direct_estimate = self.μ[TREATMENT] - self.μ[CONTROL]
    
    return A, πA, reward_estimate, direct_estimate
end 
