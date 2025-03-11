module Estimators

using ..AdaptiveATE: CONTROL, TREATMENT

export ope_linear_functional_estimator, ate_weights

"""
    ate_weights(As::Vector{Int})

Create weights for Average Treatment Effect estimation.
Returns +1 for treatment actions and -1 for control actions.
"""
function ate_weights(As::Vector{Int})
    return [A == TREATMENT ? 1.0 : -1.0 for A in As]
end

"""
    ope_linear_functional_estimator(weights::Vector{Float64}, πs::Vector{Float64}, As::Vector{Int}, Rs::Vector{Float64}, rhats::Vector{Float64}, deltas::Vector{Float64})

Compute a weighted AIPW estimate for a linear functional.

# Arguments
- `weights::Vector{Float64}`: Weight function values for each observation
- `πs::Vector{Float64}`: Propensity scores (probabilities) for selecting action 2 under the behavior policy
- `As::Vector{Int}`: Observed actions, where 1 represents control and 2 represents treatment
- `Rs::Vector{Float64}`: Observed rewards for each action
- `rhats::Vector{Float64}`: Predicted rewards for each observation
- `deltas::Vector{Float64}`: Predicted treatment effects for each observation

# Returns
- `Float64`: The weighted AIPW estimate

# Details
The weighted AIPW estimator is calculated as:
```
(1/n) * Σ weights[i] * (1/π[A[i]] * (R[i] - rhats[i]) + deltas[i])
```

# Throws
- `DimensionMismatch`: If any of the input vectors have different lengths
"""
function ope_linear_functional_estimator(weights::Vector{Float64}, πs::Vector{Float64}, As::Vector{Int}, Rs::Vector{Float64}, rhats::Vector{Float64}, deltas::Vector{Float64})
    n = length(weights)
    if any(length.([πs, As, Rs, rhats, deltas]) .!= n)
        throw(DimensionMismatch("All input vectors must have the same length"))
    end

    acc = 0.0
    @inbounds for i in 1:n

        # Compute weighted AIPW term
        acc += (weights[i] / πs[i]) * (Rs[i] - rhats[i]) + deltas[i]
    end
    return acc / n
end

end  # module Estimator