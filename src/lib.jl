module AdaptiveATE

const CONTROL = 1
const TREATMENT = 2

# Include all the base modules inside the module definition
include("utils/module.jl")
include("envs/module.jl")
include("algs/module.jl")
include("estimators.jl")
include("sim.jl")



# Re-export all the submodules
using .Utils
using .Environments
using .Algorithms
using .Estimators
using .Sim

export Algorithms, Environments, Estimators, Sim, Utils, CONTROL, TREATMENT

end