"""
    ClipOGD <: IPWAlgorithm

Clipped Online Gradient Descent algorithm for bandit problems.

# Fields
- `t::Int`: Current time step
- `m̂²::Vector{Float64}`: Estimated second moments for each arm
- `π̂::Float64`: Current probability of selecting arm 2
- `η::Float64`: Learning rate
- `α::Float64`: Exploration parameter
- `G::Float64`: Previous gradient estimate
"""
mutable struct ClipOGD <: IPWAlgorithm
    t::Int
    m̂²::Vector{Float64}
    π̂::Float64
    G::Float64
    η::Float64
    α::Float64
end

"""
    ClipOGD(η::Float64, α::Float64)

Construct a ClipOGD algorithm with specified learning rate and exploration parameter.

# Arguments
- `η::Float64`: Learning rate
- `α::Float64`: Exploration parameter
"""
function ClipOGD(η::Float64, α::Float64)
    return ClipOGD(0, zeros(2), 0.5, η, α, 0)
end

"""
    ClipOGD(T::Int)

Construct a ClipOGD algorithm with default parameters based on horizon T.

# Arguments
- `T::Int`: Time horizon
"""
function ClipOGD(T::Int)
    η = 1 / sqrt(T)
    α = sqrt(5.0 * log(T))
    return ClipOGD(0, zeros(2), 0.5, η, α, 0)
end

function update!(self::ClipOGD, A::Int, R::Real)
    self.t += 1
    δ = 0.5 * self.t^(- 1.0 / self.α)
    π̂_temp = self.π̂
    self.π̂ = clip(self.π̂ - self.η * self.G, δ, 1 - δ) # G is G_{t-1} from previous iteration


    # Compute G_t (which will be G_{t-1} for the next iteration)
    if A == 1
        self.G = R^2 / (1 - π̂_temp)^3 
    elseif A == 2
        self.G = - R^2 / π̂_temp^3
    end

end

function reset!(self::ClipOGD)
    self.t = 0
    fill!(self.m̂², 0.0)
    self.π̂ = 0.5
    self.G = 0.0
end

