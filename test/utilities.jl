n  = 1000
x  = range(0, 1, length = n)
atol = 1e-6

f(x) = x <= 0.5 ? 2x : 2x - 1

@testset "auc trapezoidal" begin
    @test auc_trapezoidal(x, x) ≈ 0.5  atol = atol
    @test auc_trapezoidal(x, x./2) ≈ 0.25 atol = atol
    @test auc_trapezoidal(x, f.(x)) ≈ 0.5  atol = atol
end
