Simulations for ClipSMT

Code for reproducing results in the paper "Logarithmic Neyman Regret for Adaptive Estimation of the Average Treatment Effect"

To run the code, you need to install the following Julia packages:

```
Julia Packages:
import Pkg;
Pkg.add("Distributions");
Pkg.add("Serialization");
Pkg.add("Distributed");
Pkg.add("IterTools");
Pkg.add("ProgressMeter")
Pkg.add("StaticArrays")
```

Then you can run `julia -t <num_threads> alg_comparison.jl` to run the simulations.
`clipping_exponent_experiments.jl` recreates the plots for the clipping exponent ratios.
`example.jl` shows how to use the package to run simulations.
