abstract type AbstractCurve end

logrange(x1, x2; kwargs...) = exp10.(range(log10(x1), log10(x2); kwargs...))

# by default always compute auc and curve points from all possible points
function auc(C::Type{<:AbstractCurve}, args...; npoints = -1, kwargs...)
    return auc_trapezoidal(apply(C, args...; npoints = npoints, kwargs...))
end

apply(C::Type{<:AbstractCurve}, cs::CMVector, args...; kwargs...) = apply(C, cs)

function apply(
    C::Type{<:AbstractCurve},
    targets::AbstractArray,
    scores::AbstractArray,
    args...;
    kwargs...)

    return apply(C, current_encoding(), targets, scores, args...; kwargs...)
end

function apply(
    C::Type{<:AbstractCurve},
    enc::TwoClassEncoding,
    targets::AbstractArray,
    scores::AbstractArray,
    args...;
    kwargs...
)

    return map(targets, scores) do t, s
        apply(C, enc, t, s, args...; kwargs...)
    end
end

function apply(
    C::Type{<:AbstractCurve},
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector;
    npoints::Int = 300,
    xscale::Symbol = :identity,
    xlims = sampling_lims(C, enc, targets),
    kwargs...
)

    # maximal resolution
    if npoints >= length(targets) + 1 || npoints <= 0
        thres = thresholds(scores)
    else
        if xscale === :identity
            quantils = range(xlims[1], xlims[2], length = npoints)
        else
            quantils = logrange(xlims[1], xlims[2], length = npoints)
        end
        thres = sampling_function(C)(enc, targets, scores, quantils)
    end
    return apply(C, enc, targets, scores, thres)
end

function apply(
    C::Type{<:AbstractCurve},
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    thres::RealVector;
    kwargs...
)

if !(0 < sum(ispositive.(enc, targets)) < length(targets))
        throw(ArgumentError("Only one class present in `targets` with encoding $enc."))
    end
    return apply(C, ConfusionMatrix(enc, targets, scores, thres))
end

# predefined functions
sampling_lims(::Type{<:AbstractCurve}, args...) = (0, 1)


# ROC curve
abstract type ROCCurve <: AbstractCurve end

"""
    roccurve(cms)
    roccurve(targets, scores; kwargs...)
    roccurve(enc, targets, scores; kwargs...)
    roccurve(targets, scores, thres; kwargs...)
    roccurve(enc, targets, scores, thres; kwargs...)

Return a roc curve (false positive rates and true positive rates) computed from the target vector and the vector of scores of directly from the vector of `ConfusionMatrix` instances.

# Arguments

- `cms::CMVector`: `ConfusionMatrix` instances
- `targets::AbstractVector`: ground truth (correct) target values
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `thres::RealVector`: a sorted vector of decision thresholds
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Keyword arguments

- `npoints::Int`: number of discretization points
- `xscale::Symbol`: scale of the x-axis (`:identity` or `:log10`)
- `xlims::Tuple{Real, Real}`: limits on the x-axis

# Examples

Basic usage of `roccurve` with default label encoding

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> scores = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> roccurve(targets, scores)
([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.0], [1.0, 0.8333333333333334, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 0.5, 0.5, 0.16666666666666666, 0.0, 0.0])

julia> roccurve(targets, scores, [0.5])
([0.5], [0.6666666666666666])

julia> roccurve(targets, scores; npoints = 3)
([0.25, 0.5, 1.0], [0.5, 0.6666666666666666, 0.6666666666666666])
```

or with custom label encoding

```jldoctest
julia> enc = OneVsOne("1", "0")
OneVsOne{String}
  positive class: 1
  negative class: 0

julia> targets = ["0", "1", "1", "0", "1", "0", "1", "1", "0", "1"];

julia> scores = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> roccurve(enc, targets, scores)
([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.0], [1.0, 0.8333333333333334, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 0.5, 0.5, 0.16666666666666666, 0.0, 0.0])

julia> roccurve(enc, targets, scores, [0.5])
([0.5], [0.6666666666666666])

julia> roccurve(enc, targets, scores; npoints = 3)
([0.25, 0.5, 1.0], [0.5, 0.6666666666666666, 0.6666666666666666])
```
"""
roccurve(args...; kwargs...) = apply(ROCCurve, args...; kwargs...)

"""
    au_roccurve(args...; kwargs...)

Return an area under the roc curve. See [`roccurve`](@ref) for more details about input arguments.
"""
au_roccurve(args...; kwargs...) = auc(ROCCurve, args...; kwargs...)
apply(::Type{ROCCurve}, cms::CMVector) = (false_positive_rate(cms), true_positive_rate(cms))

function sampling_lims(
    ::Type{ROCCurve},
    enc::TwoClassEncoding,
    targets::AbstractVector
)

    return (1/sum(isnegative.(enc, targets)), 1)
end

sampling_function(::Type{ROCCurve}) = threshold_at_fpr

# Precision-Recall curve
abstract type PRCurve <: AbstractCurve end

"""
    prcurve(args...; kwargs...)

Returns recalls and precisions.


    prcurve(cms)
    prcurve(targets, scores; kwargs...)
    prcurve(enc, targets, scores; kwargs...)
    prcurve(targets, scores, thres; kwargs...)
    prcurve(enc, targets, scores, thres; kwargs...)

Return a precision-recall curve computed from the target vector and the vector of scores of directly from the vector of `ConfusionMatrix` instances.

# Arguments

- `cms::CMVector`: `ConfusionMatrix` instances
- `targets::AbstractVector`: ground truth (correct) target values
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `thres::RealVector`: a sorted vector of decision thresholds
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Keyword arguments

- `npoints::Int`: number of discretization points
- `xscale::Symbol`: scale of the x-axis (`:identity` or `:log10`)
- `xlims::Tuple{Real, Real}`: limits on the x-axis

# Examples

Basic usage of `prcurve` with default label encoding

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> scores = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> prcurve(targets, scores)
([1.0, 0.8333333333333334, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 0.5, 0.5, 0.16666666666666666, 0.0, 0.0], [0.6, 0.5555555555555556, 0.5, 0.5, 0.6666666666666666, 0.6, 0.75, 0.75, 0.5, 0.0, 1.0])

julia> prcurve(targets, scores, [0.5])
([0.6666666666666666], [0.6666666666666666])

julia> prcurve(targets, scores; npoints = 3)
([0.5, 0.6666666666666666, 1.0], [0.75, 0.6666666666666666, 0.6])
```

or with custom label encoding

```jldoctest
julia> enc = OneVsOne("1", "0")
OneVsOne{String}
  positive class: 1
  negative class: 0

julia> targets = ["0", "1", "1", "0", "1", "0", "1", "1", "0", "1"];

julia> scores = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> prcurve(enc, targets, scores)
([1.0, 0.8333333333333334, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 0.5, 0.5, 0.16666666666666666, 0.0, 0.0], [0.6, 0.5555555555555556, 0.5, 0.5, 0.6666666666666666, 0.6, 0.75, 0.75, 0.5, 0.0, 1.0])

julia> prcurve(enc, targets, scores, [0.5])
([0.6666666666666666], [0.6666666666666666])

julia> prcurve(enc, targets, scores; npoints = 3)
([0.5, 0.6666666666666666, 1.0], [0.75, 0.6666666666666666, 0.6])
```
"""
prcurve(args...; kwargs...) = apply(PRCurve, args...; kwargs...)

"""
    au_prcurve(args...; kwargs...)

Return an area under the precision-recall curve. See [`prcurve`](@ref) for more details about input arguments.
"""
au_prcurve(args...; kwargs...) = auc(PRCurve, args...; kwargs...)
apply(::Type{PRCurve}, cms::CMVector) = (recall(cms), precision(cms))

# TODO provide a better estimate of these smallest possible TPR
function sampling_lims(
    ::Type{PRCurve},
    enc::TwoClassEncoding,
    targets::AbstractVector
)

    return (1/sum(ispositive.(enc, targets)), 1)
end

sampling_function(::Type{PRCurve}) = threshold_at_tpr
