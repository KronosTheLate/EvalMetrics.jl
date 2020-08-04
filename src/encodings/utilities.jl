default_type(ts::Type...) = default_type(promote_type(ts...))
default_type(::Type{T}) where {T} = T
default_type(::Type{<:Number}) = Float64

label(x) = x
label(x::AbstractVector) = x[1]

recode(enc::AbstractEncoding, enc_new::AbstractEncoding, x) =
    _recode(enc, enc_new, x)

function _recode(enc::TwoClassEncoding, enc_new::TwoClassEncoding, x)
    return ispositive(enc, x) ? label(positives(enc_new)) : label(negatives(enc_new))
end

function Broadcast.broadcasted(::typeof(recode), enc, enc_new, x)
    return broadcast(_recode, Ref(enc), Ref(enc_new), x)
end

classify(enc::TwoClassEncoding, score, thres) = _classify(enc, score, thres)

function Broadcast.broadcasted(::typeof(classify), enc, score, thres)
    return broadcast(_classify, Ref(enc), score, thres)
end

function _classify(enc::TwoClassEncoding, score, thres)
    return score .>= thres ? label(positives(enc)) : label(negatives(enc))
end
