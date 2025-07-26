using StaticArrays
using ..AdaptiveATE: CONTROL, TREATMENT
using Distributions

mutable struct OPT <: AIPWAlgorithm
    t::Float64  # Time step
    μ::MVector{2, Float64}  # Means
    σ²::MVector{2, Float64}  # Variances
    N::MVector{2, Float64}   # Number of pulls per arm
    π̂::Float64  # Current allocation
    δ::Float64  # Confidence parameter
    σCS::MMatrix{2, 2, Float64}  # Confidence Set for σ: [arm, bound] where bound=1 is lower, bound=2 is upper
    πCS::MVector{2, Float64}  # Confidence Set for π: [1] is lower, [2] is upper
end

function OPT(δ::Float64=0.1)
    if δ <= 0.0 || δ >= 1.0
        error("Confidence parameter must be between 0 and 1")
    end
    σCS = MMatrix{2,2}(0.0, 0.0, Inf, Inf)  # Initialize with [0,0; Inf,Inf]
    return OPT(
        0.0,
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        0.5,
        δ,
        σCS,
        @MVector[0.0, 1.0]  # πCS initialized to [0.0, 1.0]
    )
end

function compute_ci(σ::Float64, N::Float64, δ::Float64)
    if N < 1
        return (0.0, Inf)
    end
    # Using a more conservative radius based on both sample size and current estimate
    radius = sqrt(2 / N)
    return max(0.0, σ - radius), min(1.0, σ + radius)
end

function lower_ci(σ::Float64, N::Float64, δ::Float64)
  return clip(σ - 1.7 * sqrt(log(log(N) - log(δ)))/sqrt(N), 0.0, 0.5)
end

function upper_ci(σ::Float64, N::Float64, δ::Float64)
    return clip(σ + 4.2 * sqrt(log(log(N) - log(δ)))/sqrt(N), 0.0, 0.5)
end

function update!(self::OPT, A::Int, R::Real)
    @inbounds begin
        # Update formulas for maintaining running mean and variance
        μA = self.μ[A]  # Store old mean
        self.μ[A] = (self.N[A] * self.μ[A] + R) / (self.N[A] + 1.0)  # Update mean
        self.N[A] += 1.0  # Increment count after using old N for mean update
        self.σ²[A] += (1.0 / self.N[A]) * ((R - μA) * (R - self.μ[A]) - self.σ²[A])

        self.σCS[A, 1] = lower_ci(sqrt(self.σ²[A]), self.N[A], self.δ)
        self.σCS[A, 2] = upper_ci(sqrt(self.σ²[A]), self.N[A], self.δ)

        self.t += 1.0

        # Compute confidence interval for π* = σ₁/(σ₀ + σ₁)
        # Lower bound achieved when σ₁ is at its lower and σ₀ at its upper
        self.πCS[1] = self.σCS[TREATMENT, 1] / (self.σCS[CONTROL, 2] + self.σCS[TREATMENT, 1])
        # Upper bound achieved when σ₁ is at its upper and σ₀ at its lower
        self.πCS[2] = self.σCS[TREATMENT, 2] / (self.σCS[CONTROL, 1] + self.σCS[TREATMENT, 2])

        # Ensure π̂ stays within [0,1]
        if self.πCS[1] >= 0.5
            self.π̂ = clip(self.πCS[1], 0.0, 1.0)
        elseif self.πCS[2] <= 0.5
            self.π̂ = clip(self.πCS[2], 0.0, 1.0)
        else
            self.π̂ = 0.5
        end
    end
end

function reset!(self::OPT)
    @inbounds begin
        self.t = 0
        self.π̂ = 0.5
        fill!(self.μ, 0.0)
        fill!(self.σ², 0.0)
        fill!(self.N, 0.0)
        self.σCS[1:2, 1] .= 0.0  # Lower bounds
        self.σCS[1:2, 2] .= Inf  # Upper bounds
        self.πCS[1] = 0.0  # Lower bound
        self.πCS[2] = 1.0  # Upper bound
    end
end

function decide(self::OPT, context=nothing)
    A = rand(Bernoulli(self.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? self.π̂ : 1 - self.π̂
    
    # Get reward estimate as empirical mean of selected arm
    reward_estimate = self.μ[A]
    
    # Get direct estimate as difference in empirical means
    direct_estimate = self.μ[TREATMENT] - self.μ[CONTROL]
    
    return A, πA, reward_estimate, direct_estimate
end

# struct SIMDClipSMT <: BanditAlgorithm
#     t::Int
#     m̂²::Array{Float64, 2}
#     π̂::Vector{Float64}
#     cₜ::Function
#     SIMD_WIDTH::Int
# end

# function SIMDClipSMT(cₜ::Function, SIMD_WIDTH::Int)

#     return SIMDClipSMT(0, zeros(2, SIMD_WIDTH), zeros(SIMD_WIDTH), cₜ, SIMD_WIDTH)
# end

# function update!(self::SIMDClipSMT, A::Vector{Int}, R::Vector{Float64})
#     self.t += 1
#     self.m̂²[A, :] = streaming_mean(self.m̂²[A, :], self.t, R.^2)
#     cₜ = self.cₜ(self.t)
#     self.π̂ = clip.(neyman_allocation(sqrt.(self.m̂²)), cₜ, 1 - cₜ)
# end 
