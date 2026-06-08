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
# Dark-midpoint diverging: neutral -> black (blends with the background, i.e.
# "doesn't matter"), red = IS, blue = IS-NOT -- both pop. `thresh` zeroes out
# near-zero (redundant) pixels so they don't speckle the map.
const SIGNED_CMAP = cgrad([:dodgerblue, :black, :red])
function signed!(ax, v; thresh=0.12)
    m = maximum(abs, v); m = m == 0 ? 1.0 : m
    w = map(t -> abs(t) < thresh * m ? 0.0 : t, v)
    heatmap!(ax, img(w), colormap=SIGNED_CMAP, colorrange=(-m, m)); hidedecorations!(ax); hidespines!(ax)
end
# Red-only "what supports the class" view (drops the IS-NOT / off-template signal).
support!(ax, v) = (heatmap!(ax, img(max.(0.0, v)), colormap=:hot); hidedecorations!(ax); hidespines!(ax))

# instance: a 7
i7 = findfirst(i -> yte[i]==Int8(7) && predict(tm, xte[i])==Int8(7), eachindex(xte))
x = xte[i7]
sg = saliency(tm, x; occlude=:off)                          # signed: is(red) vs not(blue)
sv = [let (p,n)=Tsetlin.vote(tm, tm.clauses[k], x); p-n end for k in eachindex(tm.clauses)]
pred = argmax(sv); runner = argmax([j==pred ? typemin(Int) : sv[j] for j in eachindex(sv)])
cls = tm.classes[pred]; rcls = tm.classes[runner]
sc_vs = saliency(tm, x; target=cls, versus=rcls, occlude=:off)  # why cls and not rcls

set_theme!(theme_black())
fig = Figure(size=(1700, 900))
ax1 = Axis(fig[1,1], aspect=DataAspect(), title="input (a $(cls))"); heatmap!(ax1, img([x[i] for i in 1:length(x)]), colormap=:grays); hidedecorations!(ax1); hidespines!(ax1)
ax2 = Axis(fig[1,2], aspect=DataAspect(), title="signed: IS (red) / IS-NOT (blue)"); signed!(ax2, sg)
ax3 = Axis(fig[1,3], aspect=DataAspect(), title="support only (what makes it a $(cls))"); support!(ax3, sg)
ax4 = Axis(fig[1,4], aspect=DataAspect(), title="contrastive: why $(cls) not $(rcls)"); signed!(ax4, sc_vs)

# per-class discriminative signed: class evidence minus cross-class average ->
# red = distinctive of this digit (IS), blue = territory of other digits (IS NOT)
Label(fig[2,1:4], "Per-class discriminative saliency  —  red = IS this digit, blue = IS NOT", fontsize=20)
maps = [class_saliency(tm, [xte[i] for i in eachindex(xte) if yte[i]==Int8(d)], Int8(d); occlude=:off, limit=25)[1] for d in 0:9]
mn = mean(maps)
for d in 0:9
    ax = Axis(fig[3 + d÷5, d%5+1], aspect=DataAspect(), title=string(d)); signed!(ax, maps[d+1] .- mn)
end
Label(fig[0,:], "Signed & contrastive saliency (red = IS, dark = doesn't matter, blue = IS NOT)", fontsize=24)
p = joinpath(OUT, "xai_contrastive.png"); CairoMakie.save(p, fig); println("saved: $p")
