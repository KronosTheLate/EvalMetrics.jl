using Base: ImmutableDict

abstract type AbstractConfusionMatrix{I<:Integer} <: AbstractMatrix{I} end

struct ConfusionMatrix{N,T,I<:Integer} <: AbstractConfusionMatrix{I}
    classes::NTuple{N,T}
    data::Matrix{I}

    function ConfusionMatrix(classes)
        N = length(classes)
        cls = (classes...,)
        return new{N,eltype(cls),Int}(cls, zeros(Int, N, N))
    end
end

struct BinaryConfusionMatrix{N,T,I<:Integer} <: AbstractConfusionMatrix{I}
    classes::NTuple{N,T}
    data::Matrix{I}

    function BinaryConfusionMatrix(classes)
        if length(classes) != 2
            throw(DimensionMismatch("BinaryConfusionMatrix is defined only for 2D problems: got $(length(classes))D input. Use ConfusionMatrix instead."))
        end
        cls = (classes...,)
        return new{2,eltype(cls),Int}(cls, zeros(Int, 2, 2))
    end
end

function Base.show(io::IO, ::MIME"text/plain", C::AbstractConfusionMatrix)
    N = size(C.data, 2)
    cls = string.(C.classes)
    return pretty_table(
        io,
        Any[[cls...] C.data];
        title = summary(C),
        linebreaks = true,
        header = (
            ["Predictions ŷ →" cls...],
            ["Targets y ↓" repeat([""], N)...],
        ),
        vlines = [0, 1, N + 1],
        alignment = :c,
        crop = :both,
        vcrop_mode = :middle,
        display_size = displaysize(io),
        crop_num_lines_at_beginning = 2,
        header_crayon = Crayon(bold = true, foreground = :blue),
        subheader_crayon = Crayon(bold = true, foreground = :white),
        highlighters = (
            Highlighter(
                f = (data, i, j) -> j == 1,
                crayon = Crayon(bold = true, foreground = :white)
            ),
            Highlighter(
                f = (data, i, j) -> (i + 1) == j,
                crayon = Crayon(bold = true, foreground = :green)
            ),
        ),
        newline_at_end = false
    )
end

Base.size(C::AbstractConfusionMatrix) = size(C.data)
Base.zero(C::ConfusionMatrix) = ConfusionMatrix(C.classes)
Base.zero(C::BinaryConfusionMatrix) = BinaryConfusionMatrix(C.classes)
Base.getindex(C::AbstractConfusionMatrix, I::Vararg{Int,2}) = getindex(C.data, I...)
Base.setindex!(C::AbstractConfusionMatrix, v, I::Vararg{Int,2}) = setindex!(C.data, v, I...)

function Base.:(+)(C1::AbstractConfusionMatrix, C2::AbstractConfusionMatrix)
    cls1 = C1.classes
    cls2 = C2.classes

    if cls1 == cls2
        C = zero(C1)
        C.data .+= C1.data .+ C2.data
        return C
    elseif issubset(cls1, cls2) && issubset(cls2, cls1)
        prm = [findfirst(isequal(i), cls2) for i in cls1]
        C = zero(C1)
        C.data .+= C1.data .+ C2.data[prm, prm]
        return C
    else
        error("Confusion matrices do not have same classes.")
    end
end

labelmap(C::AbstractConfusionMatrix) = labelmap(C.classes)
labelmap(classes) = ImmutableDict(Pair.(classes, eachindex(classes))...)

function confusion(
    y,
    ŷ,
    classes = sort(unique(y));
    binary::Bool = length(classes) == 2,
    target_transform = identity,
    predict_transform = identity
)

    if !allunique(classes)
        throw(ArgumentError("Classes must be unique."))
    end
    if length(y) != length(ŷ)
        throw(DimensionMismatch("Inconsistent lengths of y and ŷ."))
    end
    C = binary ? BinaryConfusionMatrix(classes) : ConfusionMatrix(classes)
    mapping = labelmap(C)

    for (y_i, ŷ_i) in zip(y, ŷ)
        i = mapping[target_transform(y_i)]
        j = mapping[predict_transform(ŷ_i)]
        @inbounds C[i, j] += 1
    end
    return C
end

function confusion(
    y::AbstractMatrix,
    ŷ::AbstractMatrix;
    obsdim = 2,
    target_transform = argmax,
    predict_transform = argmax,
    kwargs...
)

    return confusion(
        eachslice(y; dims = obsdim),
        eachslice(ŷ; dims = obsdim),
        1:size(y, obsdim == 1 ? 2 : 1);
        target_transform,
        predict_transform,
        kwargs...
    )
end