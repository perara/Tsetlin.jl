# Shapley attribution on MNIST vs single-bit occlusion saliency, with the
# deletion/insertion faithfulness comparison. Images -> repo .cache/.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
import Pkg
for p in ("MLDatasets", "CairoMakie"); Base.find_package(p) === nothing && Pkg.add(p); end
using Statistics, Random, Printf
include(joinpath(@__DIR__, "..", "..", "src", "utils", "explain.jl"))
using .Tsetlin
using MLDatasets: MNIST
using CairoMakie

const OUT = joinpath(@__DIR__, "..", "..", ".cache"); mkpath(OUT)
EP = parse(Int, get(ENV, "XAI_EPOCHS", "30"))
SAMPLES = parse(Int, get(ENV, "XAI_SHAP_SAMPLES", "100"))

xtr, ytr = Tsetlin.unzip([MNIST(:train)...]); xte, yte = Tsetlin.unzip([MNIST(:test)...])
xtr = [Tsetlin.booleanize(x, 0.2) for x in xtr]; xte = [Tsetlin.booleanize(x, 0.2) for x in xte]
ytr = Int8.(ytr); yte = Int8.(yte)
tm = TMClassifier(xtr[1], ytr, 20, 20, 400, 150, 75, states_num=256, include_limit=128)
println("training ($EP epochs)..."); train!(tm, xtr, ytr, xte, yte, EP; verbose=0)
println("accuracy: ", round(accuracy(predict(tm, xte), yte)*100, digits=2), "%")

img(v) = reverse(reshape(Float64.(v), 28, 28), dims=2)

set_theme!(theme_black())
fig = Figure(size=(1500, 360))
for (col, i) in enumerate(findall(k -> predict(tm, xte[k])==yte[k], eachindex(xte))[1:1])
    x = xte[i]
    occ = saliency(tm, x; occlude=:off)
    shp = shapley(tm, x; samples=SAMPLES, rng=MersenneTwister(1))
    fo = faithfulness(tm, x, sortperm(occ, rev=true)); fsh = faithfulness(tm, x, sortperm(shp, rev=true))
    @printf("digit %d: faithfulness deletion AUC  occlusion %.2f | shapley %.2f\n", yte[i], fo.deletion_auc, fsh.deletion_auc)
    @printf("          faithfulness insertion AUC occlusion %.2f | shapley %.2f\n", fo.insertion_auc, fsh.insertion_auc)
    a1=Axis(fig[1,1],aspect=DataAspect(),title="input ($(yte[i]))"); heatmap!(a1, img([x[j] for j in 1:length(x)]), colormap=:grays); hidedecorations!(a1);hidespines!(a1)
    a2=Axis(fig[1,2],aspect=DataAspect(),title="occlusion saliency"); heatmap!(a2, img(max.(0.0,occ)), colormap=:hot); hidedecorations!(a2);hidespines!(a2)
    a3=Axis(fig[1,3],aspect=DataAspect(),title="Shapley ($SAMPLES samples)"); heatmap!(a3, img(max.(0.0,shp)), colormap=:hot); hidedecorations!(a3);hidespines!(a3)
    m=maximum(abs,shp); m=m==0 ? 1.0 : m
    sw = map(t -> abs(t) < 0.12*m ? 0.0 : t, shp)   # dark-midpoint + near-zero threshold
    a4=Axis(fig[1,4],aspect=DataAspect(),title="Shapley signed (IS red / IS-NOT blue)"); heatmap!(a4, img(sw), colormap=cgrad([:dodgerblue,:black,:red]), colorrange=(-m,m)); hidedecorations!(a4);hidespines!(a4)
end
Label(fig[0,:], "Shapley vs occlusion (fairer credit under feature interactions)", fontsize=22)
p = joinpath(OUT, "xai_shapley.png"); CairoMakie.save(p, fig); println("saved: $p")
