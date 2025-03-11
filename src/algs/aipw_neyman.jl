using StaticArrays
using ..AdaptiveATE: CONTROL, TREATMENT
using Distributions

"""
    OnlineAIPWNeymanAllocation <: AIPWAlgorithm

AIPW algorithm that estimates rewards online and uses Neyman allocation based on standard deviations.

# Fields
- `t::Float64`: Current time step
- `μ::MVector{2, Float64}`: Empirical means
- `σ²::MVector{2, Float64}`: Empirical variances
- `N::MVector{2, Float64}`: Number of pulls per arm
- `π̂::Float64`: Current allocation probability
"""
mutable struct OnlineAIPWNeymanAllocation <: AIPWAlgorithm
    t::Float64
    μ::MVector{2, Float64}  # Empirical means
    σ²::MVector{2, Float64}  # Empirical variances
    N::MVector{2, Float64}   # Number of pulls per arm
    π̂::Float64
end

function OnlineAIPWNeymanAllocation()
    return OnlineAIPWNeymanAllocation(
        0.0,
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        0.5
    )
end

"""
    TrueAIPWNeymanAllocation <: AIPWAlgorithm

AIPW algorithm that uses true reward function and Neyman allocation based on standard deviations.

# Fields
- `t::Float64`: Current time step
- `μ::Vector{Float64}`: True means
- `σ²::Vector{Float64}`: True variances
- `π̂::Float64`: Current allocation probability
"""
mutable struct TrueAIPWNeymanAllocation <: AIPWAlgorithm
    t::Float64
    μ::Vector{Float64}  # True means
    σ²::Vector{Float64}  # True variances
    π̂::Float64
end

function TrueAIPWNeymanAllocation(σ²::Vector{Float64}, μ::Vector{Float64})
    if length(σ²) != 2 || length(μ) != 2
        error("Variance and mean vectors must be of length 2")
    end
    # Initialize with optimal Neyman allocation based on true standard deviations
    π̂ = neyman_allocation(sqrt.(σ²))
    return TrueAIPWNeymanAllocation(0.0, μ, σ², π̂)
end

function decide(self::Union{OnlineAIPWNeymanAllocation, TrueAIPWNeymanAllocation}, context=nothing)
    A = rand(Bernoulli(self.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? self.π̂ : 1 - self.π̂
    
    # Get reward estimate as empirical/true mean of selected arm
    reward_estimate = self.μ[A]
    
    # Get direct estimate as difference in means
    direct_estimate = self.μ[TREATMENT] - self.μ[CONTROL]
    
    return A, πA, reward_estimate, direct_estimate
end

function update!(self::OnlineAIPWNeymanAllocation, A::Int, R::Real)
    @inbounds begin
        # Update formulas for maintaining running mean and variance
        μA = self.μ[A]  # Store old mean
        self.μ[A] = (self.N[A] * self.μ[A] + R) / (self.N[A] + 1.0)  # Update mean
        self.N[A] += 1.0  # Increment count after using old N for mean update
        self.σ²[A] += (1.0 / self.N[A]) * ((R - μA) * (R - self.μ[A]) - self.σ²[A])
        self.t += 1.0

        # Avoid division by zero
        if iszero(self.σ²) 
            self.π̂ = 0.5
            return
        end

        # Update allocation using Neyman allocation with standard deviations
        self.π̂ = neyman_allocation(sqrt.(self.σ²))
    end
end

function update!(self::TrueAIPWNeymanAllocation, A::Int, R::Real)
    # Only increment time step since we're using true parameters
    self.t += 1.0
end

function reset!(self::OnlineAIPWNeymanAllocation)
    @inbounds begin
        self.t = 0.0
        self.π̂ = 0.5
        fill!(self.μ, 0.0)
        fill!(self.σ², 0.0)
        fill!(self.N, 0.0)
    end
end

function reset!(self::TrueAIPWNeymanAllocation)
    self.t = 0.0
    # Don't reset π̂, μ, or σ² since they're fixed/true parameters
end 