"""
    Sim

A module for running bandit simulations, both single-threaded and multi-threaded.
This module provides functionality to simulate bandit algorithms in various environments
and estimate average treatment effects.
"""
module Sim

using ..AdaptiveATE: CONTROL, TREATMENT
using ..Algorithms: OPEAlgorithm, decide, update!, reset!, ClipSMT
using ..Environments: BanditEnv, step
using ..Estimators: ope_linear_functional_estimator, ate_weights
using ..Utils: poly_decay

using Distributed
using Base.Threads

export simulate

"""
    simulate(env::BanditEnv, algorithm::OPEAlgorithm, num_rounds::Int; 
            weight_fn=ate_weights,
            include_estimate::Bool=true,
            include_data::Bool=true,
            num_sims::Int=1,
            parallel::Bool=false)

Run bandit simulations with configurable options for data collection and parallelization.

# Arguments
- `env::BanditEnv`: The bandit environment to simulate in
- `algorithm::OPEAlgorithm`: The bandit algorithm to use
- `num_rounds::Int`: Number of rounds to simulate
- `weight_fn`: Function that takes actions vector and returns weights (default: ate_weights)
- `include_estimate`: Whether to include the estimate in the return value (default: true)
- `include_data`: Whether to include the raw data in the return value (default: true)
- `num_sims`: Number of simulations to run (default: 1)
- `parallel`: Whether to run simulations in parallel (default: false)

# Returns
For single simulation (num_sims=1):
- Dictionary containing simulation results with keys:
  - "rewards": Vector of rewards
  - "πs": Vector of action probabilities
  - "actions": Vector of actions
  - "r̂s": Vector of reward estimates
  - "Δ̂s": Vector of direct estimates
  - "estimate": ATE estimate (if include_estimate=true)

For multiple simulations:
- Vector of the above dictionaries, one per simulation
"""
function simulate(env::BanditEnv, algorithm::OPEAlgorithm, num_rounds::Int; 
                 weight_fn=ate_weights,
                 include_estimate::Bool=true,
                 include_data::Bool=true,
                 num_sims::Int=1,
                 parallel::Bool=false)
    
    num_rounds > 0 || throw(ArgumentError("num_rounds must be positive"))
    num_sims > 0 || throw(ArgumentError("num_sims must be positive"))
    include_estimate || include_data || throw(ArgumentError("At least one of include_estimate or include_data must be true"))

    # Single simulation case
    if num_sims == 1
        return _single_simulation(env, algorithm, num_rounds, weight_fn, include_estimate, include_data)
    end

    # Multiple simulations case
    if parallel
        return _parallel_simulation(env, algorithm, num_rounds, num_sims, weight_fn, include_estimate, include_data)
    else
        results = [_single_simulation(env, deepcopy(algorithm), num_rounds, weight_fn, include_estimate, include_data) for _ in 1:num_sims]
        return results
    end
end

function _single_simulation(env::BanditEnv, algorithm::OPEAlgorithm, num_rounds::Int, weight_fn, include_estimate::Bool, include_data::Bool)
    reset!(algorithm)
    rewards = Vector{Float64}(undef, num_rounds)
    πAs = Vector{Float64}(undef, num_rounds)
    As = Vector{Int}(undef, num_rounds)
    r̂s = Vector{Float64}(undef, num_rounds)
    Δ̂s = Vector{Float64}(undef, num_rounds)

    for t in 1:num_rounds
        A, πA, r̂, Δ̂ = decide(algorithm)
        R = step(env, A)
        update!(algorithm, A, R)
        rewards[t], πAs[t], As[t], r̂s[t], Δ̂s[t] = R, πA, A, r̂, Δ̂
    end

    result = Dict{String, Any}()
    
    if include_data
        result["rewards"] = rewards
        result["πs"] =  [As[t] == TREATMENT ? πAs[t] : 1 - πAs[t] for t in 1:num_rounds]
        result["actions"] = As
        result["r̂s"] = r̂s
        result["Δ̂s"] = Δ̂s
    end
    
    if include_estimate
        weights = weight_fn(As)
        result["estimate"] = ope_linear_functional_estimator(weights, πAs, As, rewards, r̂s, Δ̂s)
    end

    return result
end

function _parallel_simulation(env::BanditEnv, algorithm::OPEAlgorithm, num_rounds::Int, num_sims::Int, weight_fn, include_estimate::Bool, include_data::Bool)
    results = Vector{Any}(undef, num_sims)
    local_algorithms = [deepcopy(algorithm) for _ in 1:Threads.nthreads()]
    
    Threads.@threads for n in 1:num_sims
        thread_id = Threads.threadid()
        results[n] = _single_simulation(env, local_algorithms[thread_id], num_rounds, weight_fn, include_estimate, include_data)
    end

    return results
end

end
