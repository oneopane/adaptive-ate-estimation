"""
    BernoulliBandit <: BanditEnv

A bandit environment where each arm follows a Bernoulli distribution.

# Fields
- `p::Vector{Float64}`: Vector of success probabilities for each arm
- `K::Int`: Number of arms
"""
struct BernoulliBandit <: BanditEnv
    p::Vector{Float64}
    K::Int
end

"""
    BernoulliBandit(p::Vector{Float64})

Construct a BernoulliBandit environment with given success probabilities.

# Arguments
- `p::Vector{Float64}`: Vector of success probabilities for each arm

# Throws
- `ArgumentError`: If p is empty or contains values outside [0,1]

# Example
"""
function BernoulliBandit(p::Vector{Float64})
    if length(p) == 0
        throw(ArgumentError("p must have at least one element"))
    elseif any(x -> x < 0.0 || x > 1.0, p)
        throw(ArgumentError("p must be between 0.0 and 1.0"))
    end
    
    return BernoulliBandit(p, length(p))
end

"""
    step(env::BernoulliBandit, A::Int)

Take a step in the environment by pulling the specified arm.

# Arguments
- `env::BernoulliBandit`: The bandit environment
- `A::Int`: The arm to pull (1 to K)

# Returns
- `Float64`: Reward (0 or 1) from pulling the arm

# Throws
- `ArgumentError`: If action is not between 1 and K
"""
function step(env::BernoulliBandit, A::Int)
    if A < 1 || A > env.K
        throw(ArgumentError("action must be between 1 and $(env.K)"))
    end
    return rand(Bernoulli(env.p[A]))
end 