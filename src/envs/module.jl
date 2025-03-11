module Environments

using Distributions

export BanditEnv, BernoulliBandit, step, step_simd

abstract type BanditEnv end

include("bernoulli.jl")

end 