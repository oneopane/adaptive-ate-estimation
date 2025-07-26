module Utils

using Distributions
using Serialization
using StaticArrays

export clip, poly_decay, exp_decay, streaming_mean, neyman_allocation, compute_control_mean

function new_dir(dir::String)
    final_dir = dir
    counter = 1
    while isdir(final_dir)
        final_dir = "$(dir)_$(counter)"
        counter += 1
    end
    mkpath(final_dir)
    return final_dir
end

include("math.jl")

end  # module Utils