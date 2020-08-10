default_type(ts::Type...) = default_type(promote_type(ts...))
default_type(::Type{T}) where {T} = T
default_type(::Type{<:Number}) = Float64

label(x) = x
label(x::AbstractVector) = x[1]


"""
    recode(enc::AbstractEncoding, enc_new::AbstractEncoding, x)

Recode given value `x` encoded using label encoding `enc` to new label encoding `enc_new`.

# Examples

```jldoctest
julia> recode(OneZero(), OneVsOne(:positive, :negative), 0)
:negative

julia> recode(OneZero(), OneVsOne(:positive, :negative), 1)
:positive

julia> recode.(OneZero(), OneVsOne(:positive, :negative), [0,1,0,0,1])
5-element Array{Symbol,1}:
 :negative
 :positive
 :negative
 :negative
 :positive
```
"""
recode(enc::AbstractEncoding, enc_new::AbstractEncoding, x) = _recode(enc, enc_new, x)

function _recode(enc::TwoClassEncoding, enc_new::TwoClassEncoding, x)
    return ispositive(enc, x) ? label(positives(enc_new)) : label(negatives(enc_new))
end

function Broadcast.broadcasted(::typeof(recode), enc, enc_new, x)
    return broadcast(_recode, Ref(enc), Ref(enc_new), x)
end

"""
    classify(enc::TwoClassEncoding, score, thres)

Return a label that represents a positive class in `enc` if `score> = thres`, and a label that represents a negative class in `enc` otherwise.

# Examples

```jldoctest
julia> classify(OneZero(), 0.4, 0.5)
0

julia> classify(OneZero(), 0.6, 0.5)
1

julia> classify.(OneVsOne(:positive, :negative), [0.4, 0.6], 0.5)
2-element Array{Symbol,1}:
 :negative
 :positive
```
"""
classify(enc::TwoClassEncoding, score, thres) = _classify(enc, score, thres)

function _classify(enc::TwoClassEncoding, score, thres)
    return score .>= thres ? label(positives(enc)) : label(negatives(enc))
end

function Broadcast.broadcasted(::typeof(classify), enc, score, thres)
    return broadcast(_classify, Ref(enc), score, thres)
end
