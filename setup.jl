#!/usr/bin/env julia

using Pkg

println("Installing packages for adaptive-ate-estimation...")

packages = [
    "Distributions",
    "Serialization",
    "Distributed",
    "IterTools",
    "ProgressMeter",
    "StaticArrays",
    "Plots",
    "CSV",
    "DataFrames"
]

for package in packages
    println("Installing $package...")
    Pkg.add(package)
end

println("\nAll packages installed successfully!")
println("You can now run the examples with:")
println("  julia example.jl")
println("  julia alg_comparisons.jl")