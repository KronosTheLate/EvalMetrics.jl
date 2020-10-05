# series recipe
@recipe function f(
    ::Type{Val{:mlcurve}},
    x,
    y,
    z;
    highlight = (x,y) -> Int[],
    diagonal = false
)

    # main curve
    @series begin
        seriestype := :path
        marker := :none
        x := x
        y := y
        ()
    end

    # points on the main curve
    indexes = highlight(x,y) |> skipmissing |> collect
    if !isempty(indexes)
        indexes = filter(ind -> ind <= length(x), indexes)
        lbls = map(indexes) do i
            lbl = "($(round(x[i]; digits = 2)),  $(round(y[i]; digits = 2)))"
            return (x[i], y[i], Plots.text(lbl, :bottom, 8))
        end

        @series begin
            primary := false
            seriestype := :scatter
            markerstrokecolor := :auto
            fill := false
            label := ""
            annotations := lbls
            x := x[indexes]
            y := y[indexes]
            ()
        end
    end

    # diagonal
    if diagonal && get(plotattributes, :xscale, :identity) === :identity
        @series begin
            primary := false
            seriestype := :path
            fill := false
            line := (:red, :dash, 0.5)
            marker := :none
            label := ""
            x := [0, 1]
            y := [0, 1]
            ()
        end
    end
end

@shorthands mlcurve

"""
    mlcurve(x, y; highlight, diagonal)

Make a simple line plot of `y` vs. `x`. Both `x`, `y` should be from the interval `[0,1]`.

# Keyword arguments

- `diagonal::Bool = false`: if `true` the diagonal line from `[0,0,]` to `[1,1]` is created
- `highlight::Function = (x,y) -> Int[]`: a function that returns the indexes of the points to be highlighted
"""
mlcurve


function findrates(x, y, rates::AbstractArray; kwargs...)
    return [findrates(x, y, rate; kwargs...) for rate in rates]
end

function findrates(x, y, rate::Real; xaxis::Bool = true)
    vals = xaxis ? x : y
    ind = findfirst(val -> val >= rate, vals)
    return typeof(ind) <: Integer ? ind : missing
end

# label with auc
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

# plot limits
positives(x::Real) = x > 0 ? x : typemax(x)
minimum_pos(x) = minimum(positives, x)
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

# type recipies
@recipe function f(C::Type{<:AbstractCurve}, args...)
    points = apply(C, args...; plotattributes...)
    xlims --> _lims(points, plotattributes, :xscale)
    ylims --> _lims(points, plotattributes, :yscale)
    label := auc_label(plotattributes, auc(C, args...), args...)
    delete!(plotattributes, :npoints)
    delete!(plotattributes, :aucshow)
    return points
end

@recipe function f(C::Type{<:AbstractCurve}, cs::AbstractArray{<:CMVector}, args...)
    points = apply.(C, cs)
    xlims --> _lims(points, plotattributes, :xscale)
    ylims --> _lims(points, plotattributes, :yscale)
    label := auc_label(plotattributes, auc_trapezoidal(points), args...)
    delete!(plotattributes, :aucshow)
    return points
end

# user recipies
@userplot ROCPlot

@recipe function f(h::ROCPlot)
    seriestype := :mlcurve
    diagonal   --> true
    legend     --> :bottomright
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "ROC curve"
    xguide     --> "false positive rate"
    yguide     --> "true positive rate"
    xgrid      --> true
    ygrid      --> true
    aucshow    --> true

    (ROCCurve, h.args...)
end

"""
    rocplot(args...; kwargs...)

Make an roc curve plot. See [`roccurve`](@ref) and [`mlcurve`](@ref EvalMetrics.mlcurve) for more details about input arguments.
"""
rocplot

@userplot PRPlot

@recipe function f(h::PRPlot)
    seriestype := :mlcurve
    legend     --> :bottomleft
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "Precision-Recall curve"
    xguide     --> "recall"
    yguide     --> "precision"
    xgrid      --> true
    ygrid      --> true
    aucshow    --> true

    (PRCurve, h.args...)
end

"""
    prplot(args...; kwargs...)

Make an precision-recall curve plot. See [`prcurve`](@ref) and [`mlcurve`](@ref EvalMetrics.mlcurve) for more details about input arguments.
"""
prplot
