using ..AdaptiveATE: CONTROL, TREATMENT


@inline function clip(value::Real, lower::Real, upper::Real)
    return min(max(value, lower), upper)
end 

@inline function poly_decay(t::Int, α::Real)
    return  1 / 2 * t^(-α)
end

@inline function exp_decay(t::Int, α::Real)
    return exp(-α * t)
end 

@inline function streaming_mean(μ̂::Real, t::Int, X::Real)
    return (t * μ̂ + X) / (t + 1)
end


@inline function neyman_allocation(m::Vector{Float64})
    return m[2] / (m[1] + m[2])
end

@inline function neyman_allocation(m::MVector{2, Float64})
    return m[2] / (m[1] + m[2])
end

@inline function neyman_allocation(m::Vector{Float64}) # Can use with second moments or variances
    return m[TREATMENT] / (m[CONTROL] + m[TREATMENT])
end

@inline function neyman_allocation(m::MVector{2, Float64}) # Can use with second moments or variances
    return m[TREATMENT] / (m[CONTROL] + m[TREATMENT])
end

function nonasymptotic_neyman_allocation(σ²::Vector{Float64}, T::Int)
    
    # Define the expression to minimize
    f(pi) = (1 / T) * (σ²[TREATMENT] / pi + σ²[CONTROL] / (1 - pi)) +
            (log(T) / T^2) * ((σ²[TREATMENT] / pi) * (1 / pi - 1) + (σ²[CONTROL] / (1 - pi)) * (1 / (1 - pi) - 1))

    # Define bounds for \( \pi \)
    lower_bound = 1e-6  # Avoid division by zero
    upper_bound = 1 - 1e-6

    # Optimize \( \pi \) within bounds
    result = optimize(f, lower_bound, upper_bound)

    return Optim.minimizer(result), Optim.minimum(result)

end



@inline function compute_control_mean(treatment_mean::Float64, neyman_allocation::Float64)
    return treatment_mean * (1 - neyman_allocation)^2 / neyman_allocation^2
end
