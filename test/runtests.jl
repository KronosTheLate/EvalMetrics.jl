using Test, EvalMetrics, Random
using EvalMetrics.Encodings

@testset "EvalMetrics" begin
    include("encodings.jl")
    include("utilities.jl")
    include("confusion_matrix.jl")
    include("metrics.jl")
    include("thresholds.jl")
    include("curves.jl")
end
