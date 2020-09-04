```@setup plots
using EvalMetrics
using Plots
using Random
using MLBase
Random.seed!(42)

mkpath("./figures")

reset_encoding()
```

```@setup curves
using EvalMetrics

reset_encoding()
```


# Evaluation curves


Functionality for measuring performance with curves is implemented in the package as well. For example, a precision-recall (PR) curve can be computed as follows:
```@repl curves
scores = [0.74, 0.48, 0.23, 0.91, 0.33, 0.92, 0.83, 0.61, 0.68, 0.09]
targets = collect(1:10 .>= 3)
prcurve(targets, scores)
```

All possible calls:
- `prcurve(targets::AbstractVector, scores::RealVector)` returns all `length(target) + 1` points
- `prcurve(enc::AbstractEncoding, target::AbstractVector, scores::RealVector)` makes different encodings possible
- `prcurve(targets::AbstractVector, scores::RealVector, thres::RealVector)` uses provided threshols to compute individual points
- `prcurve(enc::AbstractEncoding, target::AbstractVector, scores::RealVector, thres::RealVector)`
- `prcurve(cms::AbstractVector{<:ConfusionMatrix})`

We can also compute area under the curve using the `auc_trapezoidal` function which uses the trapezoidal rule as follows:

```@repl curves
auc_trapezoidal(prcurve(targets, scores)...)
```

However, a convenience function `au_prcurve` is provided with exactly the same signature as `prcurve` function. Moreover, any `curve(PRCurve, args...)` or `auc(PRCurve, args...)` call is equivalent to `prcurve(args...)` and `au_prcurve(args...)`, respectively.

Besides PR curve, Receiver operating characteristic (ROC) curve is also available out of the box with analogical definitions of `roccurve` and `au_roccurve`.

All points of the curve, as well as area under curve scores are computed using the highest possible resolution by default. This can be changed by a keyword argument `npoints`

```@repl curves
length.(prcurve(targets, scores))
length.(prcurve(targets, scores; npoints=9))
auprcurve(targets, scores)
au_prcurve(targets, scores; npoints=9)
```

## Plotting
For plotting purposes, `EvalMetrics.jl` provides recipes for the `Plots` library:

```@example plots
Random.seed!(42)

scores = sort(rand(10000))
targets = scores .>= 0.99;
targets[MLBase.sample(findall(0.98 .<= scores .< 0.99), 30; replace = false)] .= true
targets[MLBase.sample(findall(0.99 .<= scores .< 0.995), 30; replace = false)] .= false
nothing # hide
```

Then, any of the following can be used:

- `prplot(targets::AbstractVector, scores::RealVector)` to use the full resolution:

```@example plots
prplot(targets, scores)
savefig("./figures/pr1.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/pr1.png?raw=true">
</p>
```

- `prplot(targets::AbstractVector, scores::RealVector, thresholds::RealVector)` to specify thresholds that will be used
- `prplot!(enc::AbstractEncoding, targets::AbstractVector, scores::RealVector)` to use a different encoding than default
- `prplot!(enc::AbstractEncoding, targets::AbstractVector, scores::RealVector, thresholds::RealVector)`

Furthermore, one can use vectors of vectors like `[targets1, targets2]` and `[scores1, scores2])` to plot multiple curves at once. The calls stay the same:

```@example plots
prplot([targets, targets], [scores, scores .+ rand(10000) ./ 5])
savefig("./figures/pr2.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/pr2.png?raw=true">
</p>
```

For ROC curve use `rocplot` analogically:

```@example plots
rocplot(targets, scores)
savefig("./figures/roc1.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/roc1.png?raw=true">
</p>
```

```@example plots
rocplot([targets, targets], [scores, scores .+ rand(10000) ./ 5])
savefig("./figures/roc2.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/roc2.png?raw=true">
</p>
```


'Modifying' versions with exclamation marks `prplot!` and `rocplot!` work as well.

The appearance of the plot can be changed in exactly the same way as with `Plots` library. Therefore, keyword arguments such as `xguide`, `xlims`, `grid`, `fill` can all be used:

```@example plots
prplot(targets, scores; xguide="RECALL", fill=:green, grid=false, xlims=(0.8, 1.0))
savefig("./figures/pr3.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/pr3.png?raw=true">
</p>
```

```@example plots
rocplot(targets, scores, title="Title", label="experiment", xscale=:log10)
savefig("./figures/roc3.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/roc3.png?raw=true">
</p>
```

Here, limits on x axis are appropriately changed, unless overridden by using `xlims` keyword argument.

```@example plots
rocplot([targets, targets], [scores, scores .+ rand(10000) ./ 5], label=["a" "b";])
savefig("./figures/roc4.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/roc4.png?raw=true">
</p>
```

By default, plotted curves have 300 points, which are sampled to retain as much information as possible. This amounts to sampling false positive rate in case of ROC curves and true positive rate in case of PR curves instead of raw thresholds. The number of points can be again changed by keyword argument `npoints`:

```@example plots
prplot(targets, scores; npoints=-1, label="Original")
prplot!(targets, scores; npoints=10, label="Sampled (10 points)")
prplot!(targets, scores; npoints=100, label="Sampled (100 points)")
prplot!(targets, scores; npoints=1000, label="Sampled (1000 points)")
prplot!(targets, scores; npoints=5000, label="Sampled (5000 points)")
savefig("./figures/roc4.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/pr4.png?raw=true">
</p>
```

Note that even though we visuallize smaller number of points, the displayed auc score is computed from all points. In case when logarithmic scale is used, the sampling is also done in logarithmic scale.

Other than that, `diagonal` keyword indicates the diagonal in the plot, and `aucshow` toggles, whether auc score is appended to a label:

```@example plots
rocplot(targets, scores; aucshow=false, label="a", diagonal=true)
savefig("./figures/roc5.png") # hide
```

```@raw html
<p align="center">
  <img src="figures/roc5.png?raw=true">
</p>
```

## User-defined curves

PR and ROC curves are available out of the box. Additional curve definitions can be provided in the similar way as new metrics are defined using macro `@curve` and defining `apply` function, which computes a point on the curve. For instance, ROC curve can be defined this way:

```@repl curves
import EvalMetrics: @curve, apply
@curve MyROCCurve
apply(::Type{MyROCCurve}, cms::AbstractVector{ConfusionMatrix{T}}) where T <: Real =
    (false_positive_rate(cms), true_positive_rate(cms))
myroccurve(targets, scores) == roccurve(targets, scores)
```

In order to be able to sample from x axis while plotting, `sampling_function` and `lowest_metric_value` must be provided as well.
