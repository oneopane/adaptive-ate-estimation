mutable struct ClipSMT <: IPWAlgorithm
    t::Float64
    m²::MVector{2, Float64}  # Changed from S² to m² (actual second moments)
    N::MVector{2, Float64}   # Number of pulls per arm
    π̂::Float64
    sqrt_means::MVector{2, Float64}  # Pre-allocated for calculations
end

function ClipSMT()
    return ClipSMT(
        0.0,
        @MVector[0.0, 0.0],
        @MVector[0.0, 0.0],
        0.5,
        @MVector[0.0, 0.0]
    )
end

function update!(self::ClipSMT, A::Int, R::Real)
    @inbounds begin
        # Update formula for maintaining running mean of squares
        self.m²[A] = (self.N[A] * self.m²[A] + R^2) / (self.N[A] + 1.0)
        self.N[A] += 1.0
        self.t += 1.0

        # Avoid division by zero
        if iszero(self.m²) 
            self.π̂ = 0.5
            return
        end

        # Now we can use m² directly without division
        @simd for i in 1:2
            self.sqrt_means[i] = sqrt(self.m²[i])
        end

        cₜ = 0.5 * self.t^(-1/3)
        self.π̂ = clip(neyman_allocation(self.sqrt_means), cₜ, 1 - cₜ)
    end
end

function reset!(self::ClipSMT)
    @inbounds begin
        self.t = 0
        self.π̂ = 0.5
        fill!(self.m², 0.0)
        fill!(self.N, 0.0)
        fill!(self.sqrt_means, 0.0)
    end
end


# struct SIMDClipSMT <: OPEAlgorithm
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