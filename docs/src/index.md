```@setup home
using EvalMetrics

reset_encoding()
```

# EvalMetrics.jl

Utility package for scoring binary classification models.


## Installation

Execute the following command in Julia Pkg REPL (`EvalMetrics.jl` requires julia 1.0 or higher)
```julia
(v1.5) pkg> add EvalMetrics
```

## Quickstart

The fastest way of getting started is to use a simple `binary_eval_report` function in the following way:

```@repl home
using EvalMetrics, Random
Random.seed!(123);

enc = current_encoding()
targets = rand(0:1, 100);
scores = rand(100);

binary_eval_report(targets, scores)
binary_eval_report(targets, scores, 0.001)
```

## Similar packages

- [MLBase.jl](https://github.com/JuliaStats/MLBase.jl):
- [MLMetrics.jl](https://github.com/JuliaML/MLMetrics.jl):
- [EvalCurves.jl](https://github.com/vitskvara/EvalCurves.jl):
- [MLJBase.jl](https://github.com/alan-turing-institute/MLJBase.jl):
- [ROC.jl](https://github.com/diegozea/ROC.jl)
- [ROCAnalysis.jl](https://github.com/davidavdav/ROCAnalysis.jl)
