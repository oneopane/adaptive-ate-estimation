# Add src/ to Julia's load path so it can find the Bandits module
include("src/lib.jl")
using .AdaptiveATE
using .Algorithms: ALG_NAMES, default_alg_constructor, IPWAlgorithm, AIPWAlgorithm
using .Environments: BernoulliBandit
using .Sim
using .Estimators: ate_weights, ope_linear_functional_estimator
using .Utils: neyman_allocation, compute_control_mean
using .AdaptiveATE: CONTROL, TREATMENT

using Distributions
using Printf
using Statistics
using Plots
using CSV
using DataFrames
using ProgressMeter

const CONFIGS = Dict(
    "large" => (num_sims=1000, num_rounds=1000:1000:20000, name="large_sample"),
    "small" => (num_sims=1000, num_rounds=100:100:3000, name="small_sample"),
    "test" => (num_sims=100, num_rounds=100:100:1000, name="test")
)

# Parse command line argument
if length(ARGS) != 1 || !(ARGS[1] in keys(CONFIGS))
    println("Usage: julia alg_comparisons.jl <config>")
    println("Available configs: ", join(keys(CONFIGS), ", "))
    exit(1)
end

# Get configuration
config = CONFIGS[ARGS[1]]
NUM_SIMS = config.num_sims
NUM_ROUNDS = config.num_rounds
NAME = config.name

# All algorithms will be run using simulations
ALGS = ["ClipSMT", "ClipOGD", "EtC", "Balanced IPW Allocation", "IPW Neyman Allocation"]

# We'll also plot theoretical lines for these algorithms
THEORETICAL_ALGS = ["Balanced IPW Allocation", "IPW Neyman Allocation"]

TREATMENT_MEANS = [0.25, 0.5, 0.75]
NEYMAN_ALLOCATIONS = [0.5, 0.6, 0.8]

# Organize means in a grid format
MEANS = [[compute_control_mean(treatment_mean, neyman_alloc), treatment_mean] 
         for neyman_alloc in NEYMAN_ALLOCATIONS 
         for treatment_mean in TREATMENT_MEANS]

results_dict = Dict()

println("Starting simulations...")
total_combinations = length(MEANS) * length(ALGS) * length(NUM_ROUNDS)
progress = Progress(total_combinations, desc="Running simulations: ", showspeed=true)

for means in MEANS
    results_dict[means] = Dict()
    
    for alg_name in ALGS
        results_dict[means][alg_name] = Dict()
        normalized_variances = Float64[]
        
        for num_rounds in NUM_ROUNDS
            env = BernoulliBandit(means)
            alg = default_alg_constructor(alg_name, num_rounds, means)    
            
            # We'll run all algorithms, including theoretical ones
            
            # Display current simulation details
            println("Running: Means=$(means), Algorithm=$(alg_name), Rounds=$(num_rounds)")
            
            results = simulate(env, alg, num_rounds, num_sims=NUM_SIMS, parallel=true, include_estimate=true, include_data=false)

            estimates = zeros(NUM_SIMS)
            for (i, result) in enumerate(results)
                estimates[i] = result["estimate"]
            end
            
            # Store estimates for all algorithms
            results_dict[means][alg_name][num_rounds] = estimates
            
            # Calculate normalized variance (variance * num_rounds)
            norm_var = var(estimates) * num_rounds
            push!(normalized_variances, norm_var)
            
            # Update progress bar
            next!(progress)
        end
    end
end

println("Simulations completed!")

# Plotting function
function create_grid_plot(results_dict, num_rounds, treatment_means, neyman_allocations)
    # Set up the grid layout
    n_rows = length(neyman_allocations)
    n_cols = length(treatment_means)
    plots = Array{Plots.Plot}(undef, n_rows, n_cols)
    
    # Color and line style mapping
    colors = Dict(
        "ClipSMT" => :blue,
        "ClipOGD" => :red,
        "EtC" => :green,
        "Balanced IPW Allocation" => :cyan,
        "IPW Neyman Allocation" => :black
    )
    
    line_styles = Dict(
        "ClipSMT" => :solid,
        "ClipOGD" => :solid,
        "EtC" => :dash,
        "Balanced IPW Allocation" => :solid,
        "IPW Neyman Allocation" => :dash
    )
    
    # Create each subplot
    for (row_idx, neyman_alloc) in enumerate(neyman_allocations)
        for (col_idx, treatment_mean) in enumerate(treatment_means)
            means = [compute_control_mean(treatment_mean, neyman_alloc), treatment_mean]
            
            # Create plot with log scale for y-axis
            # Only add labels to the top-left plot
            # Add legend to the top-left plot only
            p = plot(
                     xlabel=(row_idx==1 && col_idx==1) ? "T" : "", 
                     ylabel=(row_idx==1 && col_idx==1) ? "ATE Variance" : "", 
                     title="",
                     yscale=:log10, 
                     legend=(row_idx==1 && col_idx==1) ? :topright : false, 
                     legendfontsize=8,
                     framestyle=:box, grid=false,
                     fontfamily="Computer Modern",
                     size=(600, 300),
                     margin=0Plots.mm,
                     tickfontsize=10,
                     guidefontsize=12,
                     # Use normal notation for x ticks instead of scientific notation
                     xformatter=x->string(Int(x)))
                   
            # Add data for each algorithm from simulations
            for alg_name in ALGS
                # Extract data and calculate variance and standard error at each round
                xs = Float64[]
                ys = Float64[]
                yerrs = Float64[]
                
                for num_round in num_rounds
                    estimates = results_dict[means][alg_name][num_round]
                    v = var(estimates)
                    # Standard error of the variance for normal distribution is approximately sqrt(2/n) * variance
                    # For non-normal distributions, we use a more general formula based on fourth central moment
                    n = length(estimates)
                    m2 = sum((estimates .- mean(estimates)).^2) / n  # Second central moment (variance)
                    m4 = sum((estimates .- mean(estimates)).^4) / n  # Fourth central moment
                    se = sqrt((m4 - m2^2 * (n-1)/n) / n)  # Standard error of variance
                    
                    push!(xs, num_round)
                    push!(ys, v)
                    push!(yerrs, se)
                end
                
                # Use shorter names for the legend in the top-left plot
                legend_name = if row_idx == 1 && col_idx == 1
                    alg_name
                else
                    ""
                end
                
                # Plot the line with moderately noticeable error bars
                
                # Plot the line with error ribbon
                plot!(p, xs, ys, 
                      ribbon=yerrs,
                      fillalpha=0.3,     # Moderate opacity
                      label=legend_name, 
                      color=colors[alg_name], 
                      linestyle=line_styles[alg_name],
                      linewidth=2)
            end
            
            # # Add theoretical lines for Balanced and Neyman allocation
            # for alg_name in THEORETICAL_ALGS
            #     xs = collect(num_rounds)
            #     ys = Float64[]
                
            #     for num_round in num_rounds
            #         # Replace this with the actual variance formula provided by the user
            #         # This is just a placeholder
            #         if alg_name == "Balanced IPW Allocation"
            #             # Formula for balanced allocation variance
            #             μ_c, μ_t = means
            #             v = μ_t / 0.5 + μ_c / 0.5
            #             v = v / num_round
            #         else  # "IPW Neyman Allocation"
            #             # Formula for Neyman allocation variance
            #             μ_c, μ_t = means
            #             v = μ_t / neyman_alloc + μ_c / (1 - neyman_alloc)
            #             v = v / num_round
            #         end
                    
            #         push!(ys, v)
            #     end
                
            #     # Use shorter names for the legend in the top-left plot
            #     legend_name = if row_idx == 1 && col_idx == 1
            #         if alg_name == "Balanced IPW Allocation"
            #             "Balanced Allocation"
            #         elseif alg_name == "IPW Neyman Allocation"
            #             "Neyman Allocation"
            #         else
            #             alg_name
            #         end
            #     else
            #         ""
            #     end
                
            #     # Plot the theoretical line without error bars
            #     plot!(p, xs, ys, 
            #           label=legend_name, 
            #           color=colors[alg_name], 
            #           linestyle=line_styles[alg_name],
            #           linewidth=2)
            # end
            
            # No labels needed
            
            plots[row_idx, col_idx] = p
        end
    end
    
    # Combine all plots into a grid with minimal spacing between subplots
    # Add extra left margin to ensure y-label isn't cut off
    
    final_plot = plot(plots..., layout=(n_rows, n_cols), size=(1800, 900), 
                     margin=0Plots.mm, dpi=300, link=:both,
                     left_margin=10Plots.mm, right_margin=5Plots.mm,
                     top_margin=0Plots.mm, bottom_margin=5Plots.mm,
                     plot_titlemargin=0Plots.mm, titlefontsize=10,
                     layout_algorithm=:tight)
    
    # Legend is now in the top-left plot
    
    return final_plot
end

# Create and save the plot
final_plot = create_grid_plot(results_dict, NUM_ROUNDS, TREATMENT_MEANS, NEYMAN_ALLOCATIONS)

# Save the plot
savefig(final_plot, "figures/$(NAME)_alg_comparison.pdf")
savefig(final_plot, "figures/$(NAME)_alg_comparison.png")

println("Plots saved as $(NAME)_alg_comparison.pdf and $(NAME)_alg_comparison.png")

# Save the data used to make the plots
function save_plot_data(results_dict, num_rounds, treatment_means, neyman_allocations)
    # Create a DataFrame to store all the data
    df = DataFrame()
    
    for (row_idx, neyman_alloc) in enumerate(neyman_allocations)
        for (col_idx, treatment_mean) in enumerate(treatment_means)
            means = [compute_control_mean(treatment_mean, neyman_alloc), treatment_mean]
            
            for alg_name in ALGS
                for num_round in num_rounds
                    estimates = results_dict[means][alg_name][num_round]
                    v = var(estimates)
                    
                    # Calculate standard error of variance
                    n = length(estimates)
                    m2 = sum((estimates .- mean(estimates)).^2) / n
                    m4 = sum((estimates .- mean(estimates)).^4) / n
                    se = sqrt((m4 - m2^2 * (n-1)/n) / n)
                    
                    # Add a row to the DataFrame
                    push!(df, (
                        treatment_mean = treatment_mean,
                        control_mean = means[1],
                        neyman_allocation = neyman_alloc,
                        algorithm = alg_name,
                        num_rounds = num_round,
                        variance = v,
                        std_error = se,
                        row_idx = row_idx,
                        col_idx = col_idx
                    ))
                end
            end
        end
    end
    
    # Save the DataFrame to a CSV file
    CSV.write("data/alg_comparison_data_$(NAME).csv", df)
    println("Data saved as alg_comparison_data_$(NAME).csv")
    
    return df
end

# Save the data
save_plot_data(results_dict, NUM_ROUNDS, TREATMENT_MEANS, NEYMAN_ALLOCATIONS)
