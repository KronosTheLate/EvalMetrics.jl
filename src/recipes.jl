# TODO enable scattering points across pr and roc curves based on thresholds, tpr, fpr, or precision

# series recipe
@recipe function f(::Type{Val{:mlcurve}}, x, y, z; indexes = Int[], diagonal = false)
    # main curve
    @series begin
        seriestype := :path
        marker := :none
        x := x
        y := y
        ()
    end

    # points on the main curve
    if !isempty(indexes)
        @series begin
            primary := false
            seriestype := :scatter
            markerstrokecolor := :auto
            label := ""
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
