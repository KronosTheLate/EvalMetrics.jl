"""
    ConfusionMatrix{T<:Real}

Represenation of a confusion matrix with elements of type `T` for two-class classification problems.

# Fields

- `p::T`: number of positive samples
- `n::T`: number of negative samples
- `tp::T`: number of correctly classified positive samples
- `tn::T`: number of correctly classified negative samples
- `fp::T`: number of incorrectly classified negative samples
- `fn::T`: number of incorrectly classified positive samples
"""
struct ConfusionMatrix{T<:Real}
    p::T
    n::T
    tp::T
    tn::T
    fp::T
    fn::T

    function ConfusionMatrix(tp::Real, tn::Real, fp::Real, fn::Real)
        tp, tn, fp, fn = promote(tp, tn, fp, fn)
        return new{typeof(tp)}(tp + fn, tn + fp, tp, tn, fp, fn)
    end
end

function Base.:(+)(a::ConfusionMatrix, b::ConfusionMatrix)
    return ConfusionMatrix(a.tp + b.tp, a.tn + b.tn, a.fp + b.fp, a.fn + b.fn)
end

"""
    ConfusionMatrix(targets, predicts)
    ConfusionMatrix(enc, targets, predicts)
    ConfusionMatrix(targets, scores, thres)
    ConfusionMatrix(enc, targets, scores, thres)

Compute confusion matrix directly from

# Arguments

- `targets::AbstractVector`: a vector of targets (true labels)
- `predicts::AbstractVector`: a vector of predictions (predicted labels)
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `thres::Union{Real, RealVector}`: decision threshold or a sorted vector of decision thresholds; if `thres` is scalar the constructor returns one confusion matrix, otherwise it returns a vector of confusion matrices
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Examples

Basic usage of `ConfusionMatrix` constructor with default label encoding

```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> predicts = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> scores = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> ConfusionMatrix(targets, predicts)
ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)

julia> ConfusionMatrix(targets, scores, 0.6)
ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)

julia> ConfusionMatrix(targets, scores, [0.6, 0.6])
2-element Array{ConfusionMatrix{Int64},1}:
 ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)
 ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)
```

or with custom label encoding

```jldoctest
julia> enc = OneVsOne("1", "0")
OneVsOne{String}:
   positive class: 1
   negative class: 0

julia> targets = ["0", "1", "1", "0", "1", "0", "1", "1", "0", "1"];

julia> predicts = ["0", "1", "0", "1", "1", "0", "0", "0", "1", "1"];

julia> scores = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> ConfusionMatrix(enc, targets, predicts)
ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)

julia> ConfusionMatrix(enc, targets, scores, 0.6)
ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)

julia> ConfusionMatrix(enc, targets, scores, [0.6, 0.6])
2-element Array{ConfusionMatrix{Int64},1}:
 ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)
 ConfusionMatrix{Int64}(6, 4, 3, 2, 2, 3)
```
"""
function ConfusionMatrix(targets::AbstractVector, args...)
    return ConfusionMatrix(current_encoding(), targets, args...)
end

function ConfusionMatrix(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    predicts::AbstractVector,
)

    if length(targets) != length(predicts)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `predicts`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end
    if !check_encoding(enc, predicts)
        throw(ArgumentError("`predicts` vector uses incorrect label encoding."))
    end

    tp, tn, fp, fn = 0, 0, 0, 0

    @inbounds for i in eachindex(targets)
        if ispositive(enc, targets[i])
            ispositive(enc, predicts[i]) ? (tp += 1) : (fn += 1)
        else
            ispositive(enc, predicts[i]) ? (fp += 1) : (tn += 1)
        end
    end
    return ConfusionMatrix(tp, tn, fp, fn)
end

function ConfusionMatrix(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    thres::Real,
)

    if length(targets) != length(scores)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end

    tp, tn, fp, fn = 0, 0, 0, 0

    @inbounds for i in eachindex(targets)
        if ispositive(enc, targets[i])
            scores[i] >= thres ? (tp += 1) : (fn += 1)
        else
            scores[i] >= thres ? (fp += 1) : (tn += 1)
        end
    end
    return ConfusionMatrix(tp, tn, fp, fn)
end

function ConfusionMatrix(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    thres::RealVector,
)

    if length(targets) != length(scores)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end
    if issorted(thres)
        ts = thres
        flag_rev = false
    elseif issorted(thres; rev = true)
        ts = reverse(thres)
        flag_rev = true
    else
        throw(ArgumentError("Thresholds must be sorted."))
    end

    # scan scores and classify them into bins
    nt = length(ts)
    bins_p, bins_n = zeros(Int, nt + 1), zeros(Int, nt + 1)
    p, n = 0, 0

    @inbounds for i in eachindex(targets)
        if ispositive(enc, targets[i])
            bins_p[searchsortedlast(ts, scores[i]) + 1] += 1
            p += 1
        else
            bins_n[searchsortedlast(ts, scores[i]) + 1] += 1
            n += 1
        end
    end

    # create confusion matrices
    fn, tn = 0, 0
    cm = Array{ConfusionMatrix{Int}}(undef, nt)

    @inbounds for i in eachindex(cm)
        fn += bins_p[i]
        tn += bins_n[i]
        cm[i] = ConfusionMatrix(p - fn, tn, n - tn, fn)
    end
    return flag_rev ? reverse(cm) : cm
end
