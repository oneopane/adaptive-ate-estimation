# Add src/ to Julia's load path so it can find the Bandits module
include("src/lib.jl")
using .AdaptiveATE
using .Algorithms: ClipSMT
using .Environments: BernoulliBandit, BanditEnv
using .Sim: simulate_until_no_clipping, simulate_with_data, simulate_π_parallel
using .Utils: poly_decay, neyman_allocation
using Plots

using Distributions
using Base.Threads


function get_means(sim_number::Int)
    P = [0.5, 0.5 / sim_number]
    return P
end

# Constants
TMIN = 1000
JUMP = TMIN
TMAX = 10 * JUMP
const QUANTILE = 0.5
const Ps = [(t, get_means(t)) for t in TMIN:JUMP:TMAX]
const NUM_SIMS = 100
const MAX_ROUNDS = 1000000
const ALPHA = [collect(0.2:0.4:0.9)..., 1.0/3.0]



function compute_clipping_time(πs::Vector{Float64}, α::Float64)
    for t in length(πs):-1:1
        c = poly_decay(t, α)
        if πs[t] == c || πs[t] == (1 - c)
            return t
        end
    end
end

# function simulate_until_no_clipping_parallel(α::Float64, P::Vector{Float64}, num_sims::Int, min_rounds::Int, max_rounds::Int)
#     env = BernoulliBandit(P)
#     local_algs = [ClipSMT(α) for _ in 1:Threads.nthreads()]

#     τs = Vector{Float64}(undef, num_sims)

#     Threads.@threads for n in 1:num_sims
#        τ = simulate_with_data(env, local_algs[Threads.threadid()], MAX_ROUNDS)
#        τs[n] = τ
#     end
#     return quantile(τs, 0.5)
# end

function simulate_clipping_time(P::Vector{Float64}, α::Float64)
    env = BernoulliBandit(P)
    πs = simulate_π_parallel(env, ClipSMT(α), MAX_ROUNDS, NUM_SIMS)

    upper_quantile = [quantile([πs[i][n] for i in 1:NUM_SIMS], QUANTILE) for n in 1:MAX_ROUNDS]
    lower_quantile = [quantile([πs[i][n] for i in 1:NUM_SIMS], 1 - QUANTILE) for n in 1:MAX_ROUNDS]

    uτ = compute_clipping_time(upper_quantile, α)
    lτ = compute_clipping_time(lower_quantile, α)

    return max(uτ, lτ)
end



    
function main()    
    opt_αs =[]
    for (iP, P) in Ps

        clipping_times = Vector{Float64}(undef, length(ALPHA))
        env = BernoulliBandit(P)
        for (i, α) in enumerate(ALPHA)
            τ = simulate_clipping_time(P, α)
            clipping_times[i] = τ
        end
    
        best_idx = argmin(clipping_times)
        best_α = ALPHA[best_idx]        
        println("$iP = $P, best α = $best_α, clipping time = $(clipping_times[best_idx])")
        push!(opt_αs, best_α)
    end
    println(opt_αs)
    println([iP for (iP, P) in Ps])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end