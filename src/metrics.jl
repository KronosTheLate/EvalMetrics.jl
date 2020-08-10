abstract type AbstractMetric end

function apply(::Type{M}, args...; kwargs...) where {M <: AbstractMetric}
    return apply(M, ConfusionMatrix(args...); kwargs...)
end

function apply(
    ::Type{M},
    x::AbstractArray{<:ConfusionMatrix};
    kwargs...
) where {M <: AbstractMetric}

    return apply.(M, x; kwargs...)
end

"""
    @metric

Macro to simplify the definition of new binary classification metrics.

# Examples

Using of the macro in the following way

```julia
@metric True_positive

apply(::Type{True_positive}, x::ConfusionMatrix) = x.tp
```

is equivalent to

```julia
abstract type True_positive <: AbstractMetric end

true_positive(args...; kwargs...) = apply(True_positive, args...; kwargs...)

apply(::Type{True_positive}, x::ConfusionMatrix) = x.tp
```
"""
macro metric(name)
    name_lw = Symbol(lowercase(string(name)))

    quote
        abstract type $(esc(name)) <: AbstractMetric end

        function $(esc(name_lw))(args...; kwargs...)
            return apply($(esc(name)), args...; kwargs...)
        end
    end
end


@metric True_positive
apply(::Type{True_positive}, x::ConfusionMatrix) = x.tp

"""
    true_positive(args...)

Return the number of correctly classified positive samples. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_positive(targets, predicts)
3
```
"""
true_positive

@metric True_negative
apply(::Type{True_negative}, x::ConfusionMatrix) = x.tn

"""
    true_negative(args...)

Return the number of correctly classified negative samples. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_negative(targets, predicts)
2
```
"""
true_negative

@metric False_positive
apply(::Type{False_positive}, x::ConfusionMatrix) = x.fp

"""
    false_positive(args...)

Return the number of incorrectly classified negative samples. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_positive(targets, predicts)
2
```
"""
false_positive

@metric False_negative
apply(::Type{False_negative}, x::ConfusionMatrix) = x.fn

"""
    false_negative(args...)

Return the number of incorrectly classified positive samples. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negative(targets, predicts)
3
```
"""
false_negative

@metric True_positive_rate
apply(::Type{True_positive_rate}, x::ConfusionMatrix) = x.tp/x.p

const sensitivity = true_positive_rate
const recall      = true_positive_rate
const hit_rate    = true_positive_rate

@doc raw"""
    true_positive_rate(args...)

Return the proportion of correctly classified positive samples, i.e

```math
\mathrm{true\_positive\_rate} = \frac{tp}{p}
```

Can be also called via aliases `sensitivity`,  `recall`, `hit_rate`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negative_rate(targets, predicts)
0.5

julia> sensitivity(targets, predicts)
0.5

julia> recall(targets, predicts)
0.5

julia> hit_rate(targets, predicts)
0.5
```
"""
true_positive_rate

@metric True_negative_rate
apply(::Type{True_negative_rate}, x::ConfusionMatrix) = x.tn/x.n

const specificity = true_negative_rate
const selectivity = true_negative_rate

@doc raw"""
    true_negative_rate(args...)

Return the proportion of correctly classified positive samples, i.e

```math
\mathrm{true\_negative\_rate} = \frac{tn}{n}
```

Can be also called via aliases `specificity`,  `selectivity`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_negative_rate(targets, predicts)
0.5

julia> specificity(targets, predicts)
0.5

julia> selectivity(targets, predicts)
0.5
```
"""
true_negative_rate

@metric False_positive_rate
apply(::Type{False_positive_rate}, x::ConfusionMatrix) = x.fp/x.n

const fall_out     = false_positive_rate
const type_I_error = false_positive_rate

@doc raw"""
    false_positive_rate(args...)

Return the proportion of incorrectly classified negative samples, i.e

```math
\mathrm{false\_positive\_rate} = \frac{tn}{p}
```

Can be also called via aliases `fall_out`, `type_I_error`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_positive_rate(targets, predicts)
0.5

julia> fall_out(targets, predicts)
0.5

julia> type_I_error(targets, predicts)
0.5
```
"""
false_positive_rate

@metric False_negative_rate
apply(::Type{False_negative_rate}, x::ConfusionMatrix) = x.fn/x.p

const miss_rate     = false_negative_rate
const type_II_error = false_negative_rate

@doc raw"""
    false_negative_rate(args...)

Return the proportion of incorrectly classified positive samples, i.e

```math
\mathrm{false\_negative\_rate} = \frac{tp}{n}
```

Can be also called via aliases `miss_rate`, `type_II_error`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negative_rate(targets, predicts)
0.5

julia> miss_rate(targets, predicts)
0.5

julia> type_II_error(targets, predicts)
0.5
```
"""
false_negative_rate

@metric Precision
function apply(::Type{Precision}, x::ConfusionMatrix)
    val = x.tp/(x.tp + x.fp)
    return isnan(val) ? one(val) : val
end

const positive_predictive_value = precision

@doc raw"""
    precision(args...)

Return the ratio of positive samples in all samples classified as positive, i.e

```math
\mathrm{precision} = \frac{tp}{tp + fp}
```

Can be also called via alias `positive_predictive_value`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> precision(targets, predicts)
0.6

julia> positive_predictive_value(targets, predicts)
0.6
```
"""
precision

@metric Negative_predictive_value
apply(::Type{Negative_predictive_value}, x::ConfusionMatrix) = x.tn/(x.tn + x.fn)

@doc raw"""
    negative_predictive_value(args...)

Return the ratio of negative samples in all samples classified as positive, i.e

```math
\mathrm{negative\_predictive\_value} = \frac{tn}{tn + fn}
```

Can be also called via alias `positive_predictive_value`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negative_predictive_value(targets, predicts)
0.4
```
"""
negative_predictive_value

@metric False_discovery_rate
apply(::Type{False_discovery_rate}, x::ConfusionMatrix) = x.fp/(x.fp + x.tp)

@doc raw"""
    false_discovery_rate(args...)

Return the ratio of negative samples in all samples classified as positive, i.e

```math
\mathrm{false\_discovery\_rate} = \frac{fp}{fp + tp}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_discovery_rate(targets, predicts)
0.4
```
"""
false_discovery_rate

@metric False_omission_rate
apply(::Type{False_omission_rate}, x::ConfusionMatrix) = x.fn/(x.fn + x.tn)

@doc raw"""
    false_omission_rate(args...)

Return the ratio of positive samples in all samples classified as negatives, i.e

```math
\mathrm{false\_omission\_rate} = \frac{fn}{fn + tn}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_omission_rate(targets, predicts)
0.6
```
"""
false_omission_rate

@metric Threat_score
apply(::Type{Threat_score}, x::ConfusionMatrix) = x.tp/(x.tp + x.fn + x.fp)

const critical_success_index = threat_score

@doc raw"""
    threat_score(args...)

Return threat score defined as

```math
\mathrm{threat\_score} = \frac{tp}{tp + fn + fp}
```

Can be also called via alias `critical_success_index`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> threat_score(targets, predicts)
0.375

julia> critical_success_index(targets, predicts)
0.375
```
"""
threat_score

@metric Accuracy
apply(::Type{Accuracy}, x::ConfusionMatrix) = (x.tp + x.tn)/(x.p + x.n)

@doc raw"""
    accuracy(args...)

Return accuracy defined as

```math
\mathrm{accuracy} = \frac{tp + tn}{p + n}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> accuracy(targets, predicts)
0.5
```
"""
accuracy

@metric Balanced_accuracy
function apply(::Type{Balanced_accuracy}, x::ConfusionMatrix)
    return (true_positive_rate(x) + true_negative_rate(x))/2
end

@doc raw"""
    balanced_accuracy(args...)

Return balanced accuracy defined as

```math
\mathrm{balanced\_accuracy} = \frac{1}{2}(tpr + tnr)
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> balanced_accuracy(targets, predicts)
0.5
```
"""
balanced_accuracy

@metric Error_rate
apply(::Type{Error_rate}, x::ConfusionMatrix) = 1 - accuracy(x)

@doc raw"""
    error_rate(args...)

Return error rate defined as

```math
\mathrm{error\_rate} = 1 - \mathrm{accuracy}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> error_rate(targets, predicts)
0.5
```
"""
error_rate

@metric Balanced_error_rate
apply(::Type{Balanced_error_rate}, x::ConfusionMatrix) = 1 - balanced_accuracy(x)

@doc raw"""
    balanced_error_rate(args...)

Return balanced error rate defined as

```math
\mathrm{error\_rate} = 1 - \mathrm{balanced\_accuracy}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> balanced_error_rate(targets, predicts)
0.5
```
"""
balanced_error_rate

@metric F1_score
function apply(::Type{F1_score}, x::ConfusionMatrix)
    return 2*precision(x)*recall(x)/(precision(x) + recall(x))
end

@doc raw"""
    f1_score(args...)

Return f1 score (harmonic mean of [`precision`](@ref) and [`recall`](@ref)) defined as

```math
\mathrm{f1\_score} = 2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> f1_score(targets, predicts)
0.5454545454545454
```
"""
f1_score

@metric Fβ_score
apply(::Type{Fβ_score}, x::ConfusionMatrix; β::Real = 1) =
    (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))

@doc raw"""
    fβ_score(args...; β = 1)

Return fβ score defined as

```math
\mathrm{fβ\_score} = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> fβ_score(targets, predicts)
0.5454545454545454

julia> fβ_score(targets, predicts; β = 2)
0.5172413793103449
```
"""
fβ_score

@metric Matthews_correlation_coefficient
function apply(::Type{Matthews_correlation_coefficient}, x::ConfusionMatrix)
    return (x.tp*x.tn - x.fp*x.fn)/sqrt((x.tp + x.fp)*(x.tp + x.fn)*(x.tn + x.fp)*(x.tn + x.fn))
end

const mcc = matthews_correlation_coefficient

@doc raw"""
    matthews_correlation_coefficient(args...)

Return matthews correlation coefficient defined as

```math
\mathrm{matthews\_correlation\_coefficient} = \frac{tp \cdot tn - fp \cdot fn}{\sqrt{(tp + fp)(tp + fn)(tn + fp)(tn + fn)}}
```

Can be also called via alias `mcc`. See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> matthews_correlation_coefficient(targets, predicts)
0.0

julia> mcc(targets, predicts)
0.0
```
"""
matthews_correlation_coefficient

@metric Quant
apply(::Type{Quant}, x::ConfusionMatrix) = (x.fn + x.tn)/(x.p + x.n)

@doc raw"""
    quant(args...)

Return estimate of the quantile on classification scores that was used as a decision threshold

```math
\mathrm{quant} = \frac{fn + tn}{p + n}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> q = quant(targets, predicts)
0.5

julia> quant(targets, scores, quantile(scores, q))
0.5
```
"""
quant

@metric Topquant
apply(::Type{Topquant}, x::ConfusionMatrix) = 1 - quant(x)

@doc raw"""
    topquant(args...)

Return estimate of the top-quantile on classification scores that was used as a decision threshold

```math
\mathrm{topquant} = 1 - \mathrm{quant}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> q = topquant(targets, predicts)
0.5

julia> topquant(targets, scores, quantile(scores, q))
0.5
```
"""
topquant

@metric Positive_likelihood_ratio
function apply(::Type{Positive_likelihood_ratio}, x::ConfusionMatrix)
    return true_positive_rate(x)/false_positive_rate(x)
end

@doc raw"""
    positive_likelihood_ratio(args...)

Return positive likelihood ratio defined as

```math
\mathrm{positive\_likelihood\_ratio} = \frac{tpr}{fpr}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> positive_likelihood_ratio(targets, predicts)
1.0
```
"""
positive_likelihood_ratio

@metric Negative_likelihood_ratio
function apply(::Type{Negative_likelihood_ratio}, x::ConfusionMatrix)
    return false_negative_rate(x)/true_negative_rate(x)
end

@doc raw"""
    negative_likelihood_ratio(args...)

Return negative likelihood ratio defined as

```math
\mathrm{negative\_likelihood\_ratio} = \frac{fnr}{tnr}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negative_likelihood_ratio(targets, predicts)
1.0
```
"""
negative_likelihood_ratio

@metric Diagnostic_odds_ratio
function apply(::Type{Diagnostic_odds_ratio}, x::ConfusionMatrix)
    return true_positive_rate(x)*true_negative_rate(x)/(false_positive_rate(x)*false_negative_rate(x))
end

@doc raw"""
    diagnostic_odds_ratio(args...)

Return diagnostic odds ratio defined as

```math
\mathrm{diagnostic\_odds\_ratio} = \frac{tpr \cdot tnr}{fpr \cdot fnr}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> diagnostic_odds_ratio(targets, predicts)
1.0
```
"""
diagnostic_odds_ratio

@metric Prevalence
apply(::Type{Prevalence}, x::ConfusionMatrix) = x.p/(x.p + x.n)

@doc raw"""
    prevalence(args...)

Return prevalence defined as

```math
\mathrm{prevalence} = \frac{p}{p + n}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> prevalence(targets, predicts)
0.6
```
"""
prevalence
