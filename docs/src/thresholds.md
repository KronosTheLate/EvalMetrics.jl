# Decision thresholds for classification

The package provides a `thresholds(scores::RealVector, n::Int)` , which returns `n` decision thresholds which correspond to `n` evenly spaced quantiles of the given `scores` vector. The default value of `n` is `length(scores) + 1`.  The `thresholds` function has two keyword arguments `reduced::Bool` and `zerorecall::Bool`
- If `reduced` is `true` (default), then the function returns `min(length(scores) + 1, n)` thresholds.
- If `zerorecall`  is `true` (default), then the largest threshold is `maximum(scores)*(1 + eps())` otherwise `maximum(scores)`.

The package also provides some other useful utilities
- `threshold_at_tpr(targets::AbstractVector, scores::RealVector, tpr::Real)` returns the largest threshold `t` that satisfies `true_positive_rate(targets, scores, t) >= tpr`
- `threshold_at_tnr(targets::AbstractVector, scores::RealVector, tnr::Real)` returns the smallest threshold `t` that satisfies `true_negative_rate(targets, scores, t) >= tnr`
- `threshold_at_fpr(targets::AbstractVector, scores::RealVector, fpr::Real)` returns the smallest threshold `t` that satisfies `false_positive_rate(targets, scores, t) <= fpr`
- `threshold_at_fnr(targets::AbstractVector, scores::RealVector, fnr::Real)` returns the largest threshold `t` that satisfies `false_negative_rate(targets, scores, t) <= fnr`

All four functions can be called with an encoding of type `AbstractEncoding` as the first parameter to use a different encoding than default.