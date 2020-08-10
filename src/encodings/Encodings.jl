module Encodings

export AbstractEncoding, MultiClassEncoding, TwoClassEncoding,
       OneZero, OneMinusOne, OneTwo, OneVsOne, OneVsRest, RestVsOne,
       check_encoding, ispositive, isnegative,
       current_encoding, set_encoding, reset_encoding, recode, classify

"""
    AbstractEncoding{T}

Supertype for label encodings with labels of type `T`.

See also [`TwoClassEncoding`](@ref) and [`MultiClassEncoding`](@ref).
"""
abstract type AbstractEncoding{T} end

"""
    MultiClassEncoding{T} <: AbstractEncoding{T}

Supertype for label encodings for multi-class problems with labels of type `T`. See also [`TwoClassEncoding`](@ref) and [`MultiClassEncoding`](@ref).
"""
abstract type MultiClassEncoding{T} <: AbstractEncoding{T}; end

"""
    TwoClassEncoding{T} <: AbstractEncoding{T}

Supertype for label encodings for two-class problems with labels of type `T`.
"""
abstract type TwoClassEncoding{T} <: AbstractEncoding{T}; end

function Base.show(io::IO, enc::T) where {T <: TwoClassEncoding}
    println(io, T)
    println(io, "  positive class: ", positives(enc))
    print(io, "  negative class: ", negatives(enc))
end

include("twoclassencodings.jl")
include("utilities.jl")

const CURRENT_ENCODING = Ref{AbstractEncoding}(OneZero())

"""
    current_encoding()

Return the label encoding that is currently used as the default encoding.

See also [`set_encoding`](@ref) and [`reset_encoding`](@ref).

# Examples

```jldoctest
julia> reset_encoding()
OneZero{Int64}
  positive class: 1
  negative class: 0

julia> current_encoding()
OneZero{Int64}
  positive class: 1
  negative class: 0
```
"""
current_encoding() = CURRENT_ENCODING[]

"""
    set_encoding(enc::AbstractEncoding)

Set `end` as the default encoding.

See also [`current_encoding`](@ref) and [`reset_encoding`](@ref).

# Examples

```jldoctest
julia> set_encoding(OneVsOne(:positive, :negative))
OneVsOne{Symbol}
  positive class: positive
  negative class: negative

julia> current_encoding()
OneVsOne{Symbol}
  positive class: positive
  negative class: negative
```
"""
set_encoding(enc::AbstractEncoding) = CURRENT_ENCODING[] = enc

"""
    reset_encoding()

Set `end` the default encoding to [`OneZero`](@ref) encoding.

See also [`current_encoding`](@ref) and [`set_encoding`](@ref).

# Examples

```jldoctest
julia> reset_encoding()
OneZero{Int64}
  positive class: 1
  negative class: 0

julia> current_encoding()
OneZero{Int64}
  positive class: 1
  negative class: 0
```
"""
reset_encoding() = CURRENT_ENCODING[] = OneZero()

end
