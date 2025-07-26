# Add src/ to Julia's load path so it can find the Bandits module
include("src/lib.jl")
using .AdaptiveATE
using .Algorithms: ALG_NAMES, default_alg_constructor, IPWAlgorithm, AIPWAlgorithm
using .Environments: BernoulliBandit
using .Sim
using .Estimators: ate_weights, ope_linear_functional_estimator
using .Utils: neyman_allocation
using .AdaptiveATE: CONTROL, TREATMENT

using Distributions
using Printf
using Plots
using Statistics

# Constants
const P = [0.5, 0.5]
const TRUE_ATE = P[TREATMENT] - P[CONTROL]
const NUM_ROUNDS = 1000
const NUM_SIMS = 1000

function plot_pi_trajectories(πs_matrix, alg)
    p = plot(
        1:NUM_ROUNDS, 
        mean(πs_matrix, dims=1)', 
        ribbon=std(πs_matrix, dims=1)', 
        label="Mean π₂",
        title="Treatment Probability Trajectories",
        xlabel="Round",
        ylabel="π₂",
        legend=:bottomright,
        alpha=0.3
    )
    # Use different Neyman allocation calculations based on algorithm type
    optimal_treatment_prob = if alg isa AIPWAlgorithm
        neyman_allocation(sqrt.((P .* (1 .- P))))  # Standard deviations for AIPW
    else
        neyman_allocation(sqrt.(P))  # Square root of raw moments for IPW
    end
    hline!([optimal_treatment_prob], label="Optimal π₂", linestyle=:dash)
    return p
end

function run_all_algs()
    env = BernoulliBandit(P)
    plots = []
    all_means = []  # Store all mean values to determine y-axis limits
    all_stds = []   # Store all std values to determine y-axis limits
    
    # First pass: collect all data to determine y-axis limits
    for alg_name in ALG_NAMES
        alg = default_alg_constructor(alg_name, NUM_ROUNDS, P, P)
        
        # Parallel simulation collecting data
        elapsed_time = @elapsed results = simulate(env, alg, NUM_ROUNDS, num_sims=NUM_SIMS, parallel=true, include_data=true)
        
        # Extract π trajectories from all simulations
        πs_matrix = zeros(NUM_SIMS, NUM_ROUNDS)
        estimates = zeros(NUM_SIMS)
        
        for (i, result) in enumerate(results)
            # Convert πs to treatment probabilities based on actions
            πs = result["πs"]
            πs_matrix[i, :] = πs
            estimates[i] = result["estimate"]
        end
        
        # Store means and stds for y-axis limits
        push!(all_means, mean(πs_matrix, dims=1)[:])
        push!(all_stds, std(πs_matrix, dims=1)[:])
        
        # Compute mean and variance of (estimate - TRUE_ATE)
        estimation_errors = estimates .- TRUE_ATE
        @printf("%8.8s  ATE: %8.6f  Error Var: %8.6f  Time: %6.2f s\n", 
                alg_name, mean(estimates), var(estimation_errors), elapsed_time)
        
        # Create plot but store data for later
        push!(plots, (alg, πs_matrix))
    end
    
    # Calculate global y-axis limits
    all_means_flat = vcat(all_means...)
    all_stds_flat = vcat(all_stds...)
    global_min = minimum(all_means_flat .- 2 .* all_stds_flat)
    global_max = maximum(all_means_flat .+ 2 .* all_stds_flat)
    
    # Second pass: create plots with consistent y-axis limits
    final_plots = []
    for (alg, πs_matrix) in plots
        p = plot(
            1:NUM_ROUNDS, 
            mean(πs_matrix, dims=1)', 
            ribbon=std(πs_matrix, dims=1)', 
            label="Mean π₂",
            title="$(typeof(alg).name.name) - Treatment Probability",
            xlabel="Round",
            ylabel="π₂",
            legend=:bottomright,
            alpha=0.3,
            ylims=(global_min, global_max)
        )
        
        # Add optimal allocation line
        optimal_treatment_prob = if alg isa AIPWAlgorithm
            neyman_allocation(sqrt.((P .* (1 .- P))))  # Standard deviations for AIPW
        else
            neyman_allocation(sqrt.(P))  # Square root of raw moments for IPW
        end
        hline!([optimal_treatment_prob], label="Optimal π₂", linestyle=:dash)
        
        push!(final_plots, p)
    end
    
    # Add an empty plot to make it 3x3 with 8 plots
    push!(final_plots, plot(title="", grid=false, showaxis=false))  # Empty plot for padding
    final_plot = plot(final_plots..., layout=(3,3), size=(1800, 1800))
    savefig(final_plot, "pi_trajectories.png")
end

function run_single_alg(alg_name::String = "ClipSMT")
    env = BernoulliBandit(P)
    alg = default_alg_constructor(alg_name, NUM_ROUNDS, P, P)
    
    # Run simulation collecting data
    elapsed_time = @elapsed results = simulate(env, alg, NUM_ROUNDS, num_sims=NUM_SIMS, parallel=true, include_data=true)
    
    # Extract π trajectories from all simulations
    πs_matrix = zeros(NUM_SIMS, NUM_ROUNDS)
    estimates = zeros(NUM_SIMS)
    
    for (i, result) in enumerate(results)
        # Convert πs to treatment probabilities based on actions
        πs = result["πs"]
        actions = result["actions"]
        for t in 1:NUM_ROUNDS
            # If action was CONTROL, then π[TREATMENT] = 1 - π[A_t]
            # If action was TREATMENT, then π[TREATMENT] = π[A_t]
            πs_matrix[i, t] = actions[t] == CONTROL ? 1 - πs[t] : πs[t]
        end
        estimates[i] = result["estimate"]
    end
    
    # Compute mean and variance of (estimate - TRUE_ATE)
    estimation_errors = estimates .- TRUE_ATE
    @printf("%8.8s  ATE: %8.6f  Error Var: %8.6f  Time: %6.2f s\n", 
            alg_name, mean(estimates), var(estimation_errors), elapsed_time)
    
    # Create and save plot
    p = plot_pi_trajectories(πs_matrix, alg)
    title!(p, "$(alg_name) - Treatment Probability Trajectories")
    savefig(p, "$(lowercase(alg_name))_pi_trajectory.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_algs()
end
