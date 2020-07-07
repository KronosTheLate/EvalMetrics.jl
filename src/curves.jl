# TODO enable scattering points across pr and roc curves based on thresholds, tpr, fpr, or precision
# TODO implement curve persistance
# TODO implement multi-curve plotting
# TODO update readme with aucs and curve plotting
# TODO seamless use (thresholds and other arguments are optional)
# TODO other Plots arguments, such as log x
# TODO test everything
# TODO change readme and notify moonshot
# TODO pridat priklad od Vaska do README

abstract type AbstractCurve end


function check_targets(::Type{C}, enc::TwoClassEncoding, targets::AbstractVector) where {C<:AbstractCurve}
    if !(0 < sum(ispositive.(enc, targets)) < length(targets))
        throw(ArgumentError("Only one class present in `targets` with encoding $enc."))
    end
end


function apply(::Type{C},
               targets::AbstractVector,
               scores::RealVector,
               thres::RealVector = thresholds(scores)) where {C<:AbstractCurve}

    return apply(C, current_encoding(), targets, scores, thres)
end


function apply(::Type{C},
               enc::TwoClassEncoding,
               targets::AbstractVector,
               scores::RealVector,
               thres::RealVector = thresholds(scores))  where {C<:AbstractCurve}

    check_targets(C, enc, targets)
    return apply(C, ConfusionMatrix(enc, targets, scores, thres))
end


auc(::Type{C}, args...) where {C<:AbstractCurve} = auc_trapezoidal(curve(C, args...)...)
curve(::Type{C}, args...) where {C<:AbstractCurve} = apply(C, args...)


macro curve(name)
    name_lw = Symbol(lowercase(string(name)))
    name_auc = Symbol(lowercase(string("au_", name)))

    quote 
        abstract type $(esc(name)) <: AbstractCurve end

        Base.@__doc__  function $(esc(name_lw))(args...; kwargs...) 
            apply($(esc(name)), args...; kwargs...)
        end

        function $(esc(name_auc))(args...; kwargs...) 
            auc($(esc(name)), args...; kwargs...)
        end
    end
end


@recipe function f(::Type{Val{:mlcurve}}, x, y, z; indexes = Int[], aucshow = true, diagonal = false)

    # Set attributes
    grid  --> true
    lims  --> (0, 1.01)
    if aucshow
        user_label = get(plotattributes, :label, "AUTO")
        auc_label = string("auc: ", round(100*auc_trapezoidal(x,y), digits = 2), "%")

        if user_label != "AUTO"
            label := string(user_label, " (", auc_label, ")")
        else
            label := auc_label
        end
    end

    # main curve
    @series begin
        seriestype := :path
        marker     := :none
        x          := x
        y          := y
        ()
    end

    # points on the main curve
    if !isempty(indexes)
        @series begin
            primary           := false
            seriestype        := :scatter
            markerstrokecolor := :auto
            label             := ""
            x                 := x[indexes]
            y                 := y[indexes]
            ()
        end 
    end

    # diagonal
    if diagonal
        @series begin
            primary    := false
            seriestype := :path
            fill       := false
            line       := (:red, :dash, 0.5)
            marker     := :none
            label      := ""
            x          := [0, 1]
            y          := [0, 1]
            ()
        end 
    end
end

@shorthands mlcurve

@recipe f(::Type{C}, args...) where {C<:AbstractCurve} = apply(C, args...)
@recipe f(::Type{C}, cs::AbstractArray{<:CMVector}) where {C<:AbstractCurve} = apply.(C, cs)


# ROC curve
"""
    $(SIGNATURES) 

Returns false positive rates and true positive rates.
"""
@curve ROCCurve
apply(::Type{ROCCurve}, cms::CMVector) = (false_positive_rate(cms), true_positive_rate(cms))

@userplot ROCPlot

@recipe function f(h::ROCPlot)
    seriestype := :mlcurve
    diagonal   --> true
    legend     := :bottomright
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "ROC curve"
    xguide     --> "false positive rate"
    yguide     --> "true positive rate"

    (ROCCurve, h.args...)
end


# Precision-Recall curve
"""
    $(SIGNATURES) 

Returns recalls and precisions.
"""
@curve PRCurve
apply(::Type{PRCurve}, cms::CMVector) = (recall(cms), precision(cms)) 

@userplot PRPlot

@recipe function f(h::PRPlot)
    seriestype := :mlcurve
    legend     := :bottomleft
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "Precision-Recall curve"
    xguide     --> "recall"
    yguide     --> "precision"

    (PRCurve, h.args...)
end
