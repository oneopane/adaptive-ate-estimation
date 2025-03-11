mutable struct ClipSDT <: AIPWAlgorithm
    t::Float64
    μ::MVector{2, Float64}  # Empirical means
    σ²::MVector{2, Float64}  # Empirical variances
    N::MVector{2, Float64}   # Number of pulls per arm
    π̂::Float64
    std_devs::MVector{2, Float64}  # Pre-allocated for calculations
    α::Float64
end

function ClipSDT(α::Float64)
    if α <= 0.0 || α >= 1.0
        error("Clipping exponent must be between 0 and 1")
    end
    return ClipSDT(
        0.0,
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        0.5,
        @MVector[0.0, 0.0],
        α
    )
end

function decide(self::ClipSDT, context=nothing)
    A = rand(Bernoulli(self.π̂)) ? TREATMENT : CONTROL
    πA = A == TREATMENT ? self.π̂ : 1 - self.π̂
    
    # Get reward estimate as empirical mean of selected arm
    reward_estimate = self.μ[A]
    
    # Get direct estimate as difference in empirical means
    direct_estimate = self.μ[TREATMENT] - self.μ[CONTROL]
    
    return A, πA, reward_estimate, direct_estimate
end

function update!(self::ClipSDT, A::Int, R::Real)
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

        # Now compute standard deviations
        @simd for i in 1:2
            self.std_devs[i] = sqrt(self.σ²[i]) # max to avoid numerical issues
        end

        cₜ = 0.5 * self.t^(-self.α)
        self.π̂ = clip(neyman_allocation(self.std_devs), cₜ, 1 - cₜ)
    end
end

function reset!(self::ClipSDT)
    @inbounds begin
        self.t = 0
        self.π̂ = 0.5
        fill!(self.μ, 0.0)
        fill!(self.σ², 0.0)
        fill!(self.N, 0.0)
        fill!(self.std_devs, 0.0)
    end
end 