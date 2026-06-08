# Signed / contrastive saliency: red = evidence the input IS the class,
# blue = evidence it is NOT (toward another class). Plus "why A and not B".
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
import Pkg
for p in ("MLDatasets", "CairoMakie"); Base.find_package(p) === nothing && Pkg.add(p); end
using Statistics
include(joinpath(@__DIR__, "..", "..", "src", "utils", "explain.jl"))
using .Tsetlin
using MLDatasets: MNIST
using CairoMakie

const OUT = joinpath(@__DIR__, "..", "..", ".cache"); mkpath(OUT)
EP = parse(Int, get(ENV, "XAI_EPOCHS", "30"))

xtr, ytr = Tsetlin.unzip([MNIST(:train)...]); xte, yte = Tsetlin.unzip([MNIST(:test)...])
xtr = [Tsetlin.booleanize(x, 0.2) for x in xtr]; xte = [Tsetlin.booleanize(x, 0.2) for x in xte]
ytr = Int8.(ytr); yte = Int8.(yte)
tm = TMClassifier(xtr[1], ytr, 20, 20, 400, 150, 75, states_num=256, include_limit=128)
println("training ($EP epochs)..."); train!(tm, xtr, ytr, xte, yte, EP; verbose=0)
println("accuracy: ", round(accuracy(predict(tm, xte), yte)*100, digits=2), "%")

img(v) = reverse(reshape(Float64.(v), 28, 28), dims=2)
function diverging!(ax, v)
    m = maximum(abs, v); m = m == 0 ? 1.0 : m
    heatmap!(ax, img(v), colormap=:RdBu, colorrange=(-m, m)); hidedecorations!(ax); hidespines!(ax)
end

# instance: a 7
i7 = findfirst(i -> yte[i]==Int8(7) && predict(tm, xte[i])==Int8(7), eachindex(xte))
x = xte[i7]
sg = saliency(tm, x; occlude=:off)                          # signed: is(red) vs not(blue)
# runner-up class
sv = [let (p,n)=Tsetlin.vote(tm, tm.clauses[k], x); p-n end for k in eachindex(tm.clauses)]
pred = argmax(sv); runner = argmax([j==pred ? typemin(Int) : sv[j] for j in eachindex(sv)])
cls = tm.classes[pred]; rcls = tm.classes[runner]
sc_vs = saliency(tm, x; target=cls, versus=rcls, occlude=:off)  # why cls and not rcls

set_theme!(theme_black())
fig = Figure(size=(1500, 900))
ax1 = Axis(fig[1,1], aspect=DataAspect(), title="input (a $(cls))"); heatmap!(ax1, img([x[i] for i in 1:length(x)]), colormap=:grays); hidedecorations!(ax1); hidespines!(ax1)
ax2 = Axis(fig[1,2], aspect=DataAspect(), title="signed saliency: IS (red) / IS-NOT (blue)"); diverging!(ax2, sg)
ax3 = Axis(fig[1,3], aspect=DataAspect(), title="contrastive: why $(cls) (red) not $(rcls) (blue)"); diverging!(ax3, sc_vs)

# per-class discriminative signed: class evidence minus cross-class average ->
# red = distinctive of this digit (IS), blue = territory of other digits (IS NOT)
Label(fig[2,1:3], "Per-class discriminative saliency  —  red = IS this digit, blue = IS NOT", fontsize=20)
maps = [class_saliency(tm, [xte[i] for i in eachindex(xte) if yte[i]==Int8(d)], Int8(d); occlude=:off, limit=25)[1] for d in 0:9]
mn = mean(maps)
for d in 0:9
    ax = Axis(fig[3 + d÷5, d%5+1], aspect=DataAspect(), title=string(d)); diverging!(ax, maps[d+1] .- mn)
end
Label(fig[0,:], "Signed & contrastive saliency (what it IS vs what it is NOT)", fontsize=24)
p = joinpath(OUT, "xai_contrastive.png"); CairoMakie.save(p, fig); println("saved: $p")
