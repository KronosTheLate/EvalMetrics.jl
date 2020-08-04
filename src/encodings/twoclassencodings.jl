check_encoding(enc::AbstractEncoding, x::AbstractVector) = all(check_encoding.(enc, x))
check_encoding(enc::AbstractEncoding, x) = _check_encoding(enc, x)
_check_encoding(enc::TwoClassEncoding, x)= ispositive(enc, x) || isnegative(enc, x)

function Broadcast.broadcasted(::typeof(check_encoding), enc, x)
    return broadcast(_check_encoding, Ref(enc), x)
end

compare(x, y) = x == y
compare(x, y::AbstractVector) = x in y

ispositive(enc::TwoClassEncoding, x) = _ispositive(enc, x)
_ispositive(enc::TwoClassEncoding, x) = compare(x, positives(enc))
Broadcast.broadcasted(::typeof(ispositive), enc, x) = broadcast(_ispositive, Ref(enc), x)

isnegative(enc::TwoClassEncoding, x) = _isnegative(enc, x)
_isnegative(enc::TwoClassEncoding, x) = compare(x, negatives(enc))
Broadcast.broadcasted(::typeof(isnegative), enc, x) = broadcast(_isnegative, Ref(enc), x)

positives(enc::TwoClassEncoding) = enc.positives
negatives(enc::TwoClassEncoding) = enc.negatives

"""
    OneZero{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `zero(T)` the negative class.
"""
struct OneZero{T<:Number} <: TwoClassEncoding{T}
    OneZero(::Type{T} = Int64) where {T<:Number} = new{T}()
end

positives(::OneZero{T}) where T = one(T)
negatives(::OneZero{T}) where T = zero(T)


"""
    OneMinusOne{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `-one(T)` the negative class.
"""
struct OneMinusOne{T<:Number} <: TwoClassEncoding{T}
    OneMinusOne(::Type{T} = Int64) where {T<:Number} = new{T}()
end

positives(::OneMinusOne{T}) where T = one(T)
negatives(::OneMinusOne{T}) where T = -one(T)

"""
    OneTwo{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `2*one(T)` the negative class.
"""
struct OneTwo{T<:Number} <: TwoClassEncoding{T}
    OneTwo(::Type{T} = Int64) where {T<:Number} = new{T}()
end

positives(::OneTwo{T}) where T = one(T)
negatives(::OneTwo{T}) where T = 2*one(T)

"""
    OneVsOne{T} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct OneVsOne{T} <: TwoClassEncoding{T}
    positives::T
    negatives::T

    function OneVsOne(pos::P, neg::N) where {P, N}
        T = default_type(P, N)
        return new{T}(T(pos), T(neg))
    end
end

"""
    OneVsRest{T} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct OneVsRest{T} <: TwoClassEncoding{T}
    positives::T
    negatives::Vector{T}

    function OneVsRest(pos::P, neg::AbstractVector{N}) where {P, N}
        T = default_type(P, N)
        return new{T}(T(pos), T.(neg))
    end
end

"""
    RestVsOne{T} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct RestVsOne{T} <: TwoClassEncoding{T}
    positives::Vector{T}
    negatives::T

    function RestVsOne(pos::AbstractVector{P}, neg::N) where {P, N}
        T = default_type(P, N)
        return new{T}(T.(pos), T(neg))
    end
end
