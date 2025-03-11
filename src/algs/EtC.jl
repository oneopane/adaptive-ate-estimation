"""
    EtC <: IPWAlgorithm

Explore-then-Commit algorithm for bandit problems.

# Fields
- `t::Int`: Current time step
- `S²::Vector{Float64}`: Sum of squared rewards for each arm during exploration
- `π̂::Float64`: Current probability of selecting arm 2
- `explore_rounds::Int`: Number of rounds to explore before committing
"""
mutable struct EtC <: IPWAlgorithm
    t::Int
    S²::Vector{Float64}
    π̂::Float64
    explore_rounds::Int
end

"""
    EtC(explore_rounds::Int)

Construct an EtC algorithm with specified exploration rounds.

# Arguments
- `explore_rounds::Int`: Number of rounds to explore before committing
"""
function EtC(explore_rounds::Int)
    return EtC(0, zeros(2), 0.5, explore_rounds)
end

"""
    EtC(num_rounds::Int, alpha::Float64)

Construct an EtC algorithm with exploration rounds determined by total rounds.

# Arguments
- `num_rounds::Int`: Total number of rounds
- `alpha::Float64`: Parameter controlling exploration duration (T^alpha rounds)
"""
function EtC(num_rounds::Int, alpha::Float64)
    explore_rounds = Int(round(num_rounds^alpha))
    return EtC(0, zeros(2), 0.5, explore_rounds)
end

function update!(alg::EtC, A::Int, R::Real)
    if alg.t == alg.explore_rounds # Commit at the start of round alg.T
        if iszero(alg.S²)
            alg.π̂ = 0.5
        else
            alg.π̂ = neyman_allocation(sqrt.(alg.S² / alg.t))
        end
    elseif alg.t < alg.explore_rounds # Dont need to update m̂² after committing
        alg.S²[A] += R^2 # update
    end
    alg.t += 1
end

function reset!(self::EtC)
    self.t = 0
    self.S² = zeros(2)
    self.π̂ = 0.5
end

