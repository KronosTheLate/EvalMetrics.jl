"""
    thresholds(scores::RealVector, n::Int = length(scores))

Returns `n` decision thresholds which correspond to `n` evenly spaced quantiles
of the given vector of scores.

# Keyword arguments

- `reduced::Bool = true`: if `true` the resulting number of thresholds is
`min(length(scores), n)`
- `zerorecall::Bool = true`: if `true` the resulting number of thresholds is
`n + 1` and the largest threshold is `maximum(scores) + eps()`

# Examples

```jldoctest
julia> thresholds(0:1:3, 5; reduced = true, zerorecall = true)
5-element Array{Float64,1}:
 0.0
 1.0
 2.0
 3.0
 3.0000000000000004

julia> thresholds(0:1:3, 5; reduced = false, zerorecall = true)
6-element Array{Float64,1}:
 0.0
 0.75
 1.5
 2.25
 3.0
 3.0000000000000004

julia> thresholds(0:1:3, 5; reduced = true, zerorecall = false)
4-element Array{Float64,1}:
 0.0
 1.0
 2.0
 3.0

julia> thresholds(0:1:3, 5; reduced = false, zerorecall = false)
5-element Array{Float64,1}:
 0.0
 0.75
 1.5
 2.25
 3.0
```
"""
function thresholds(
    scores::RealVector,
    n::Int = length(scores);
    reduced::Bool = true,
    zerorecall::Bool = true,
)

    N = reduced ? min(length(scores), n) : n
    thres = quantile(scores, range(0, 1, length = N))
    if zerorecall
        return vcat(thres, nextfloat(thres[end]))
    else
        return thres
    end
end


"""
    threshold_at_k(scores::RealVector, k::Int; rev::Bool = true)

If `rev=true` return `k`-th largest score and `k`-th smallest score otherwise.

# Examples

```jldoctest
julia> threshold_at_k(0:1:10, 3)
8

julia> threshold_at_k(0:1:10, 3; rev = false)
2
```
"""
function threshold_at_k(scores::RealVector, k::Int; rev::Bool = true)

    n = length(scores)
    if n < k
        throw(ArgumentError("`k` must be smaller or equal to `length(scores) = $(n)`."))
    end
    return partialsort(scores, k, rev = rev)
end

function threshold_at_rate(scores::RealVector, rates::RealVector; rev::Bool = true)

    all(0 .<= rates .<= 1) || throw(ArgumentError("input rates out of [0, 1]."))
    issorted(rates) || throw(ArgumentError("input rates must be sorted."))

    n_rates = length(rates)
    n_scores = length(scores)
    print_warn = falses(n_rates)

    # case rate == 1
    if rev
        thresh = fill(nextfloat(scores[end]), n_rates)
        thresh[rates .== 1] .= scores[end]
    else
        thresh = fill(scores[end], n_rates)
    end

    # case rate != 1
    rate_last = 0
    t_last = scores[1]
    j = 1

    for (i, score) in enumerate(scores)
        t_last == score && continue

        # compute current rate
        rate = (i-1)/n_scores

        for rate_i in rates[j:end]
            rate <= rate_i && break
            rate_last == 0 && rate_i != 0 && (print_warn[j] = true)

            thresh[j] = rev ? nextfloat(t_last) : t_last
            j += 1
            j > n_rates && (return thresh, print_warn)
        end

        # update last rate and threshold
        rate_last = rate
        t_last = score
    end
    return thresh, print_warn
end


@doc raw"""
    threshold_at_tpr(targets, scores, α)
    threshold_at_tpr(enc, targets, scores, α)

For given true positive rate `α ∈ [0, 1]` return a decision threshold `t` defined as

```math
\sup_{t} \{t \mid tpr(targets, scores, t) \geq \alpha\}
```

where `tpr(targets, scores, t)` is the true positive rate computed for given decision threshold `t`, see [`true_positive_rate`](@ref)

# Arguments

- `targets::AbstractVector`: ground truth (correct) target values
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `α::Union{Real, RealVector}`: true positive rate or a sorted vector of true poitive rates; if `α` is scalar the function returns one threshold, otherwise it returns a vector of thresholds
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Examples
```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> t = threshold_at_tpr(targets, scores, 0.5)
0.7

julia> true_positive_rate(targets, scores, t)
0.5

julia> true_positive_rate(targets, scores, t + eps(t))
0.3333333333333333
```
"""
function threshold_at_tpr(targets::AbstractVector, scores::RealVector, α)
    return threshold_at_tpr(current_encoding(), targets, scores, α)
end


function threshold_at_tpr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::Real,
)

    return threshold_at_tpr(enc, targets, scores, [α])[1]
end


function threshold_at_tpr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::RealVector,
)

    if length(targets) != length(scores)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end

    scores_pos = sort(scores[ispositive.(enc, targets)]; rev = false)
    rates = round.(1 .- reverse(α); digits = 14)
    ts, print_warn = threshold_at_rate(scores_pos, rates; rev = false)

    if any(print_warn) && SHOW_WARN[]
        rates = α[reverse(print_warn)]
        @warn "The closest higher feasible true positive rate to some of the required values ($(join(rates, ", "))) is 1.0!"
    end
    return reverse(ts)
end

@doc raw"""
    threshold_at_tnr(targets, scores, α)
    threshold_at_tnr(enc, targets, scores, α)

For given true negative rate `α ∈ [0, 1]` return a decision threshold `t` defined as

```math
\inf_{t} \{t \mid tnr(targets, scores, t) \geq \alpha\}
```

where `tnr(targets, scores, t)` is the true negative rate computed for given decision threshold `t`, see [`true_negative_rate`](@ref)

# Arguments

- `targets::AbstractVector`: ground truth (correct) target values
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `α::Union{Real, RealVector}`: true positive rate or a sorted vector of true poitive rates; if `α` is scalar the function returns one threshold, otherwise it returns a vector of thresholds
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Examples
```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> t = threshold_at_tnr(targets, scores, 0.5)
0.4000000000000001

julia> true_negative_rate(targets, scores, t)
0.5

julia> true_negative_rate(targets, scores, t - eps(t))
0.25
```
"""
function threshold_at_tnr(targets::AbstractVector, scores::RealVector, α)
    return threshold_at_tnr(current_encoding(), targets, scores, α)
end


function threshold_at_tnr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::Real,
)

    return threshold_at_tnr(enc, targets, scores, [α])[1]
end


function threshold_at_tnr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::RealVector,
)

    if length(targets) != length(scores)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end

    scores_neg = sort(scores[isnegative.(enc, targets)]; rev = true)
    rates = round.(1 .- reverse(α); digits = 14)
    ts, print_warn = threshold_at_rate(scores_neg, rates; rev = true)

    if any(print_warn) && SHOW_WARN[]
        rates = α[reverse(print_warn)]
        @warn "The closest higher feasible true negative rate to some of the required values ($(join(rates, ", "))) is 1.0!"
    end
    return reverse(ts)
end

@doc raw"""
    threshold_at_fpr(targets, scores, α)
    threshold_at_fpr(enc, targets, scores, α)

For given false positive rate `α ∈ [0, 1]` return a decision threshold `t` defined as

```math
\inf_{t} \{t \mid fpr(targets, scores, t) \leq \alpha\}
```

where `fpr(targets, scores, t)` is the false positive rate computed for given decision threshold `t`, see [`false_positive_rate`](@ref)

# Arguments

- `targets::AbstractVector`: ground truth (correct) target values
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `α::Union{Real, RealVector}`: true positive rate or a sorted vector of true poitive rates; if `α` is scalar the function returns one threshold, otherwise it returns a vector of thresholds
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Examples
```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> t = threshold_at_fpr(targets, scores, 0.5)
0.4000000000000001

julia> false_positive_rate(targets, scores, t)
0.5

julia> false_positive_rate(targets, scores, t - eps(t))
0.75
```
"""
function threshold_at_fpr(targets::AbstractVector, scores::RealVector, α)
    return threshold_at_fpr(current_encoding(), targets, scores, α)
end


function threshold_at_fpr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::Real,
)

    return threshold_at_fpr(enc, targets, scores, [α])[1]
end

function threshold_at_fpr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::RealVector,
)

    if length(targets) != length(scores)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end

    scores_neg = sort(scores[isnegative.(enc, targets)]; rev = true)
    ts, print_warn = threshold_at_rate(scores_neg, α; rev = true)

    if any(print_warn) && SHOW_WARN[]
        rates = α[print_warn]
        @warn "The closest lower feasible false positive rate to some of the required values ($(join(rates, ", "))) is 0.0!"
    end
    return ts
end

@doc raw"""
    threshold_at_fnr(targets, scores, α)
    threshold_at_fnr(enc, targets, scores, α)

For given false negative rate `α ∈ [0, 1]` return a decision threshold `t` defined as

```math
\sup_{t} \{t \mid fnr(targets, scores, t) \leq \alpha\}
```

where `fnr(targets, scores, t)` is the false negative rate computed for given decision threshold `t`, see [`false_negative_rate`](@ref)

# Arguments

- `targets::AbstractVector`: ground truth (correct) target values
- `scores::RealVector`: a vector of scores given by the classifier; a sample `i` is classified as positive if `scores[i] >= t`, where `t` is the decision threshold
- `α::Union{Real, RealVector}`: true positive rate or a sorted vector of true poitive rates; if `α` is scalar the function returns one threshold, otherwise it returns a vector of thresholds
- `enc::TwoClassEncoding`: label encoding for two-class problems; the default encoding is given by [`current_encoding`](@ref) function

# Examples
```jldoctest
julia> targets = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> t = threshold_at_fnr(targets, scores, 0.5)
0.7

julia> false_negative_rate(targets, scores, t)
0.5

julia> false_negative_rate(targets, scores, t + eps(t))
0.6666666666666666
```
"""
function threshold_at_fnr(targets::AbstractVector, scores::RealVector, α)
    return threshold_at_fnr(current_encoding(), targets, scores, α)
end

function threshold_at_fnr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::Real,
)

    return threshold_at_fnr(enc, targets, scores, [α])[1]
end


function threshold_at_fnr(
    enc::TwoClassEncoding,
    targets::AbstractVector,
    scores::RealVector,
    α::RealVector,
)

    if length(targets) != length(scores)
        throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    end
    if !check_encoding(enc, targets)
        throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    end

    scores_pos = sort(scores[ispositive.(enc, targets)]; rev = false)
    ts, print_warn = threshold_at_rate(scores_pos, α; rev = false)

    if any(print_warn) && SHOW_WARN[]
        rates = α[print_warn]
        @warn "The closest lower feasible false negative rate to some of the required values ($(join(rates, ", "))) is 0.0!"
    end
    return ts
end
