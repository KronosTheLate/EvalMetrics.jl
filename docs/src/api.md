# API

## Confusion matrix

```@docs
ConfusionMatrix
ConfusionMatrix(targets::AbstractVector, args...)
```

## Classification metrics

```@docs
EvalMetrics.@metric
true_negative
true_positive
false_positive
false_negative
true_positive_rate
true_negative_rate
false_positive_rate
false_negative_rate
precision
negative_predictive_value
false_discovery_rate
false_omission_rate
threat_score
accuracy
balanced_accuracy
error_rate
balanced_error_rate
f1_score
fβ_score
matthews_correlation_coefficient
quant
topquant
positive_likelihood_ratio
negative_likelihood_ratio
diagnostic_odds_ratio
prevalence
```

## Curves

```@docs
EvalMetrics.@curve
ROCCurve
roccurve
rocplot
au_roc
PRCurve
prcurve
prplot
au_pr
auc_trapezoidal
```

## Decision thresholds

```@docs
thresholds
threshold_at_k
threshold_at_tpr
threshold_at_tnr
threshold_at_fpr
threshold_at_fnr
```

## Encodings

```@docs
AbstractEncoding
MultiClassEncoding
TwoClassEncoding
OneZero
OneMinusOne
OneTwo
OneVsOne
OneVsRest
RestVsOne
current_encoding
set_encoding
reset_encoding
ispositive
isnegative
recode
classify
```
