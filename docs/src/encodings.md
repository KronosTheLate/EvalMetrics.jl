# Label encodings

Different label encodings are considered common in different machine learning applications. For example, support vector machines use `1` as a positive label and `-1` as a negative label. On the other hand, it is common for neural networks to use `0` as a negative label. The package provides some basic label encodings listed in the following table

| Encoding              | positive label(s) | negative label(s) |
| :--                   | :--               | :--               |
| [`OneZero`](@ref)     | `one(T)`          | `zero(T)`         |
| [`OneMinusOne`](@ref) | `one(T)`          | `-one(T)`         |
| [`OneTwo`](@ref)      | `one(T)`          | `2*one(T)`        |
| [`OneVsOne`](@ref)    | `pos`             | `neg`             |
| [`OneVsRest`](@ref)   | `pos`             | `neg`             |
| [`RestVsOne`](@ref)   | `pos`             | `neg`             |

The [`current_encoding`](@ref) function can be used to verify which encoding is currently in use (by default it is [`OneZero`](@ref) encoding)

```@setup encodings
using EvalMetrics, Random

Random.seed!(123)
reset_encoding()
```

```@example encodings
targets = rand(0:1, 100)
predicts  = rand(100) .>= 0.6
nothing #hide
```

```@repl encodings
enc = current_encoding()
```

One way to use a different encoding is to pass the new encoding as the first argument

```@repl encodings
enc_new = OneVsOne(:positive, :negative)
targets_recoded = recode.(enc, enc_new, targets);
predicts_recoded = recode.(enc, enc_new, predicts);
recall(enc, targets, predicts)
recall(enc_new, targets_recoded, predicts_recoded)
```
The second way is to change the current encoding to the one you want

```@repl encodings
set_encoding(OneVsOne(:positive, :negative))
recall(targets_recoded, predicts_recoded)
```
