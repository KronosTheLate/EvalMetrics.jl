# Classification metrics

## Confusion Matrix

The core the package is the [`ConfusionMatrix`](@ref) structure, which represents the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) in the following form

| `` ``                   |    Actual positives    |    Actual negatives    |
|:--                      |:-:                     |:-:                     |
| **Predicted positives** | tp (# true positives)  | fp (# false positives) |
| **Predicted negatives** | fn (# false negatives) | tn (# true negatives)  |
|                         | p (# positives)        | n (# negatives)        |

The confusion matrix can be calculated from targets and predicted values or from targets, scores, and one or more decision thresholds

```@setup metrics
using EvalMetrics
import Random

Random.seed!(123)
reset_encoding()
```

```@example metrics
targets = rand(0:1, 100)
scores = rand(100)
thres = 0.6
predicts  = scores .>= thres
nothing #hide
```

```@repl metrics
cm1 = ConfusionMatrix(targets, predicts)
cm2 = ConfusionMatrix(targets, scores, thres)
cm3 = ConfusionMatrix(targets, scores, thres)
cm4 = ConfusionMatrix(targets, scores, [thres, thres])
```

The package provides many basic classification metrics based on the confusion matrix.  The following table provides a list of all available metrics and its aliases

| Classification metric                      | Aliases                              |
| :--                                        | :--                                  |
| [`true_positive`](@ref)                    |                                      |
| [`true_negative`](@ref)                    |                                      |
| [`false_positive`](@ref)                   |                                      |
| [`false_negative`](@ref)                   |                                      |
| [`true_positive_rate`](@ref)               | `sensitivity`,  `recall`, `hit_rate` |
| [`true_negative_rate`](@ref)               | `specificity`,  `selectivity`        |
| [`false_positive_rate`](@ref)              | `fall_out`, `type_I_error`           |
| [`false_negative_rate`](@ref)              | `miss_rate`, `type_II_error`         |
| [`precision`](@ref)                        | `positive_predictive_value`          |
| [`negative_predictive_value`](@ref)        |                                      |
| [`false_discovery_rate`](@ref)             |                                      |
| [`false_omission_rate`](@ref)              |                                      |
| [`threat_score`](@ref)                     | `critical_success_index`             |
| [`accuracy`](@ref)                         |                                      |
| [`balanced_accuracy`](@ref)                |                                      |
| [`error_rate`](@ref)                       |                                      |
| [`balanced_error_rate`](@ref)              |                                      |
| [`f1_score`](@ref)                         |                                      |
| [`fβ_score`](@ref)                         |                                      |
| [`matthews_correlation_coefficient`](@ref) | `mcc`                                |
| [`quant`](@ref)                            |                                      |
| [`positive_likelihood_ratio`](@ref)        |                                      |
| [`negative_likelihood_ratio`](@ref)        |                                      |
| [`diagnostic_odds_ratio`](@ref)            |                                      |
| [`prevalence`](@ref)                       |                                      |

Each metric can be computed from the [`ConfusionMatrix`](@ref) structure

```@repl metrics
recall(cm1)
recall(cm2)
recall(cm3)
recall(cm4)
```

The other option is to compute the metric directly from targets and predicted values or from targets, scores, and one or more decision thresholds

```@repl metrics
recall(targets, predicts)
recall(targets, scores, thres)
recall(targets, scores, thres)
recall(targets, scores, [thres, thres])
```

## User defined classification metrics
It may occur that some useful metric is not defined in the package. To simplify the process of defining a new metric, the package provides the [`@metric`](@ref) macro and [`apply`](@ref) function.

```@example metrics
import EvalMetrics: @metric, apply

@metric MyRecall

apply(::Type{MyRecall}, x::ConfusionMatrix) = x.tp/x.p
```

In the previous example, macro [`@metric`](@ref) defines a new abstract type `MyRecall` (used for dispatch) and a function `myrecall` (for easy use of the new metric).  With defined abstract type `MyRecall`, the next step is to define a new method for the [`apply`](@ref) function. This method must have exactly two input arguments: `Type{MyRecall}` and [`ConfusionMatrix`](@ref).  If another argument is needed, it can be added as a keyword argument.

```julia
function apply(::Type{Fβ_score}, x::ConfusionMatrix; β::Real = 1)
    return (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))
end
```

It is easy to check that the `myrecall` metric returns the same outputs as the [`recall`](@ref) metric defined in the package

```@repl metrics
myrecall(cm1)
myrecall(cm2)
myrecall(cm3)
myrecall(cm4)
myrecall(targets, predicts)
myrecall(targets, scores, thres)
myrecall(targets, scores, thres)
myrecall(targets, scores, [thres, thres])
```
