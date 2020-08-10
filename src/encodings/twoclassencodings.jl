"""
    check_encoding(enc::AbstractEncoding, x)

Return `true` if `x` is encoded using `enc` label encoding and `false` otherwise.

```jldoctest
julia> check_encoding(OneZero(), 1)
true

julia> check_encoding(OneVsOne(:positive, :negative), [:negative, :negative, :positive])
true

julia> check_encoding(OneVsOne(:positive, :negative), [0, 1, 0, 0, 1])
false
```
"""
check_encoding(enc::AbstractEncoding, x::AbstractArray) = all(check_encoding.(enc, x))
check_encoding(enc::AbstractEncoding, x) = _check_encoding(enc, x)
_check_encoding(enc::TwoClassEncoding, x)= ispositive(enc, x) || isnegative(enc, x)

function Broadcast.broadcasted(::typeof(check_encoding), enc, x)
    return broadcast(_check_encoding, Ref(enc), x)
end

compare(x, y) = x == y
compare(x, y::AbstractVector) = x in y

"""
    ispositive(enc::AbstractEncoding, x)

Return `true` if `x` represents a positive class in the `enc` label encoding and `false` otherwise.

# Examples

```jldoctest
julia> ispositive(OneZero(), 1)
true

julia> ispositive(OneVsOne(:positive, :negative), :negative)
false

julia> ispositive.(OneZero(), [0,1,0,0,1])
5-element BitArray{1}:
 0
 1
 0
 0
 1
```
"""
ispositive(enc::TwoClassEncoding, x) = _ispositive(enc, x)
_ispositive(enc::TwoClassEncoding, x) = compare(x, positives(enc))
Broadcast.broadcasted(::typeof(ispositive), enc, x) = broadcast(_ispositive, Ref(enc), x)

"""
    isnegative(enc::AbstractEncoding, x)

Return `true` if `x` represents a negative class in the `enc` label encoding and `false` otherwise.

# Examples

```jldoctest
julia> isnegative(OneZero(), 0)
true

julia> isnegative(OneVsOne(:positive, :negative), :positive)
false

julia> isnegative.(OneZero(), [0,1,0,0,1])
5-element BitArray{1}:
 1
 0
 1
 1
 0
```
"""
isnegative(enc::TwoClassEncoding, x) = _isnegative(enc, x)
_isnegative(enc::TwoClassEncoding, x) = compare(x, negatives(enc))
Broadcast.broadcasted(::typeof(isnegative), enc, x) = broadcast(_isnegative, Ref(enc), x)

positives(enc::TwoClassEncoding) = enc.positives
negatives(enc::TwoClassEncoding) = enc.negatives

"""
    OneZero{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `zero(T)` the negative class.

# Examples

```jldoctest
julia> OneZero()
OneZero{Int64}
  positive class: 1
  negative class: 0
```
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

# Examples

```jldoctest
julia> OneMinusOne()
OneMinusOne{Int64}
  positive class: 1
  negative class: -1
```
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

# Examples

```jldoctest
julia> OneTwo()
OneTwo{Int64}
  positive class: 1
  negative class: 2
```
"""
struct OneTwo{T<:Number} <: TwoClassEncoding{T}
    OneTwo(::Type{T} = Int64) where {T<:Number} = new{T}()
end

positives(::OneTwo{T}) where T = one(T)
negatives(::OneTwo{T}) where T = 2*one(T)

"""
    OneVsOne{T} <: TwoClassEncoding{T}

Two class label encoding in which positive and negative class is represented by one label.

# Examples

```jldoctest
julia> OneVsOne(1, 0)
OneVsOne{Float64}
  positive class: 1.0
  negative class: 0.0

julia> OneVsOne(:positive, :negative)
OneVsOne{Symbol}
  positive class: positive
  negative class: negative
```
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

Two class label encoding in which positive class is represented by one label and negative class is represented by multiple labels.

# Examples

```jldoctest
julia> OneVsRest(1, [2,3,4,5,6])
OneVsRest{Float64}
  positive class: 1.0
  negative class: [2.0, 3.0, 4.0, 5.0, 6.0]

julia> OneVsRest(:positive, [:negative, :unknown])
OneVsRest{Symbol}
  positive class: positive
  negative class: [:negative, :unknown]
```
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

Two class label encoding in which positive class is represented by multiple labels and negative class is represented by one label.

# Examples

```jldoctest
julia> RestVsOne([1,2,3], 4)
RestVsOne{Float64}
  positive class: [1.0, 2.0, 3.0]
  negative class: 4.0

julia> RestVsOne([:good ,:perfect], :bad)
RestVsOne{Symbol}
  positive class: [:good, :perfect]
  negative class: bad
```
"""
struct RestVsOne{T} <: TwoClassEncoding{T}
    positives::Vector{T}
    negatives::T

    function RestVsOne(pos::AbstractVector{P}, neg::N) where {P, N}
        T = default_type(P, N)
        return new{T}(T.(pos), T(neg))
    end
end
