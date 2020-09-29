n = 1000
targets = rand(0:1, n)
scores = rand(n)
thres = 0.7
predicts = scores .>= thres

tp = sum(targets .* predicts)
tn = sum((1 .- targets) .* (1 .- predicts))
fp = sum((1 .- targets) .* predicts)
fn = sum(targets .* (1 .- predicts))

cm = ConfusionMatrix(tp, tn, fp, fn)
cm2 = ConfusionMatrix(2*tp, 2*tn, 2*fp, 2*fn)

set_encoding(OneZero())

encs = [
    OneZero(),
    OneMinusOne(),
    OneTwo(),
    OneVsOne(3,4),
    OneVsOne(:three,:four),
    OneVsOne("three","four"),
    OneVsRest(1, [2,3,4]),
    OneVsRest(:one, [:two, :three]),
    OneVsRest("one", ["two", "three"]),
    RestVsOne([1,2,3], 4),
    RestVsOne([:one, :two], :three),
    RestVsOne(["one", "two"], "three")
]

shapes = [(n, ), (1, n), (1, n, 1)]

@testset "ConfusionMatrix +" begin
    @test cm + cm == cm2
end


@testset "ConfusionMatrix constructors for $enc encoding" for enc in encs
    global targets, predicts

    targets = recode.(current_encoding(), enc, targets)
    predicts = recode.(current_encoding(), enc, predicts)
    set_encoding(enc)

    @testset "shape = $(shape)" for shape in shapes
        targts = reshape(targets, shape...)
        predicts = reshape(predicts, shape...)

        @test ConfusionMatrix(targets, predicts) == cm
        @test ConfusionMatrix(enc, targets, predicts) == cm
        @test ConfusionMatrix(targets, scores, thres) == cm
        @test ConfusionMatrix(enc, targets, scores, thres) == cm
        @test ConfusionMatrix(targets, scores, [thres]) == [cm]
        @test ConfusionMatrix(enc, targets, scores, [thres]) == [cm]
    end
end
