logrange(x1, x2; kwargs...) = exp10.(range(log10(x1), log10(x2); kwargs...))
positives(x::T) where {T<:Real} = x > 0 ? x : typemax(T)
minimum_pos(x) = minimum(positives, x)

"""
    auc(x::RealVector, y::RealVector)

Computes the area under curve `(x,y)` using trapezoidal rule.
"""
function auc_trapezoidal(x::RealVector, y::RealVector)

    n = length(x)
    n == length(y) || throw(DimensionMismatch("Inconsistent lengths of `x` and `y`."))

    if issorted(x)
        prm = 1:n
    elseif issorted(x, rev = true)
        prm = n:-1:1
    else
        throw(ArgumentError("`x` must be sorted."))
    end

    val = zero(promote_type(eltype(x), eltype(y)))
    for i in 2:n
        Δx   = x[prm[i]]  - x[prm[i-1]]
        Δy   = (y[prm[i]] + y[prm[i-1]])/2
        if !(isnan(Δx) || isnan(Δy) || Δx == 0)
            val += Δx*Δy
        end
    end
    return val
end

auc_trapezoidal(tpls::AbstractArray{<:Tuple}) = auc_trapezoidal.(tpls)
auc_trapezoidal((x,y)::Tuple{RealVector, RealVector}) = auc_trapezoidal(x, y)

# by default always compute auc and curve points from all possible points
function auc(C::Type{<:AbstractCurve}, args...; npoints = -1, kwargs...)
    return auc_trapezoidal(apply(C, args...; npoints = npoints, kwargs...))
end

function auc_label(plotattributes, auc_score, args...)
    user_label = get(plotattributes, :label, "AUTO")

    if get(plotattributes, :aucshow, false)
        auc_label = string.("auc: ", round.(100 * auc_score', digits = 2), "%")
        if user_label != "AUTO"
            return string.(user_label, " (", auc_label, ")")
        else
            return auc_label
        end
    else
        user_label
    end
end

ident_lims() = (0, 1.01)
log_lims(x::Tuple, f) = (minimum_pos(f(x)), 1.01)
log_lims(x, f) = (minimum(minimum_pos, f.(x)), 1.01)


function _lims(points, plotattributes, key)
    scale = get(plotattributes, key, :identity)
    if scale == :identity
        return ident_lims()
    else
        return key == :xscale ? log_lims(points, first) : log_lims(points, last)
    end
end
