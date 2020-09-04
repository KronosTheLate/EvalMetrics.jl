abstract type AbstractCurve end

apply(C::Type{<:AbstractCurve}, cs::CMVector, args...; kwargs...) = apply(C, cs, args...)

function apply(C::Type{<:AbstractCurve}, args...; kwargs...)
    return apply(C, current_encoding(), args...;kwargs...)
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
    roccurve(args...; kwargs...)

Returns false positive rates and true positive rates.
"""
roccurve(args...; kwargs...) = apply(ROCCurve, args...; kwargs...)
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
"""
prcurve(args...; kwargs...) = apply(PRCurve, args...; kwargs...)
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
