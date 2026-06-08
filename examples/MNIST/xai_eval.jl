# Full XAI evaluation + visualization on MNIST. Images -> repo .cache/.
#   - faithfulness: occlusion vs Shapley vs random, mean +/- std, AUC labels
#   - per-instance overlays (signed saliency, counterfactual, clause rules) for a
#     CORRECT and a MISCLASSIFIED example
#   - per-class saliency; global importance; necessity-by-violation; stability diff
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
import Pkg
for p in ("MLDatasets", "CairoMakie"); Base.find_package(p) === nothing && Pkg.add(p); end
using Statistics, Random, Printf
include(joinpath(@__DIR__, "..", "..", "src", "utils", "explain.jl"))
using .Tsetlin
using MLDatasets: MNIST
using CairoMakie
include(joinpath(@__DIR__, "xai_plots.jl"))

const OUT = joinpath(@__DIR__, "..", "..", ".cache"); mkpath(OUT)
EP = parse(Int, get(ENV, "XAI_EPOCHS", "100"))
NEVAL = parse(Int, get(ENV, "XAI_NEVAL", "25"))
SHAP = parse(Int, get(ENV, "XAI_SHAP_SAMPLES", "40"))
rng = MersenneTwister(1)

xtr, ytr = Tsetlin.unzip([MNIST(:train)...]); xte, yte = Tsetlin.unzip([MNIST(:test)...])
xtr = [Tsetlin.booleanize(x, 0.2) for x in xtr]; xte = [Tsetlin.booleanize(x, 0.2) for x in xte]
ytr = Int8.(ytr); yte = Int8.(yte)
tm = TMClassifier(xtr[1], ytr, 20, 20, 400, 150, 75, states_num=256, include_limit=128)
println("training (epochs=$EP) ..."); train!(tm, xtr, ytr, xte, yte, EP; verbose=0)
println("accuracy: ", round(accuracy(predict(tm, xte), yte)*100, digits=2), "%")

tmbits(x) = Float64.([x[i] for i in 1:length(x)])
set_theme!(theme_black())

# ===================== figure 1: faithfulness (occlusion vs shapley vs random) =====================
steps = 25; xs = collect(range(0, 1; length=steps+1))
del = Dict(:occ=>Vector{Float64}[], :shap=>Vector{Float64}[], :rand=>Vector{Float64}[])
ins = Dict(:occ=>Vector{Float64}[], :shap=>Vector{Float64}[], :rand=>Vector{Float64}[])
idxs = Int[]; for i in randperm(rng, length(xte)); predict(tm, xte[i])==yte[i] && push!(idxs, i); length(idxs)>=NEVAL && break; end
for i in idxs
    x = xte[i]
    orders = (:occ=>sortperm(saliency(tm, x; occlude=:off), rev=true),
              :shap=>sortperm(shapley(tm, x; samples=SHAP, rng=rng), rev=true),
              :rand=>randperm(rng, length(x)))
    for (k, o) in orders
        f = faithfulness(tm, x, o; steps=steps); b = max(f.base_margin, 1.0)
        push!(del[k], f.deletion ./ b); push!(ins[k], f.insertion ./ b)
    end
end
auc(c) = mean(mean(v) for v in c)
function band!(ax, curves, color, label)
    M = reduce(hcat, curves); mu = vec(mean(M, dims=2)); sd = vec(std(M, dims=2))
    CairoMakie.band!(ax, xs, mu .- sd, mu .+ sd, color=(color, 0.16)); lines!(ax, xs, mu, color=color, linewidth=3, label=label)
end
fig1 = Figure(size=(1200, 480))
axd = Axis(fig1[1,1], title="Deletion (faithful = drops fast)", xlabel="fraction removed", ylabel="norm. margin")
band!(axd, del[:occ], :orange, @sprintf("occlusion (AUC %.2f)", auc(del[:occ])))
band!(axd, del[:shap], :cyan, @sprintf("shapley (AUC %.2f)", auc(del[:shap])))
band!(axd, del[:rand], :gray, @sprintf("random (AUC %.2f)", auc(del[:rand]))); axislegend(axd)
axi = Axis(fig1[1,2], title="Insertion (faithful = rises fast)", xlabel="fraction inserted", ylabel="norm. margin")
band!(axi, ins[:occ], :orange, @sprintf("occlusion (AUC %.2f)", auc(ins[:occ])))
band!(axi, ins[:shap], :cyan, @sprintf("shapley (AUC %.2f)", auc(ins[:shap])))
band!(axi, ins[:rand], :gray, @sprintf("random (AUC %.2f)", auc(ins[:rand]))); axislegend(axi, position=:rb)
Label(fig1[0,:], "Faithfulness over $(length(idxs)) instances (mean ± std)", fontsize=24)
CairoMakie.save(joinpath(OUT, "xai_faithfulness.png"), fig1); println("saved xai_faithfulness.png")

# ===================== figure 2: per-instance overlays (correct + misclassified) =====================
function top_clause_rule(x, cls)
    ex = explain(tm, x); ecs = ex[cls][true].clauses
    o = sortperm(ecs, by=c->c.vote, rev=true)
    # green = positive literals the clause required and got (digit-shaped); red =
    # violations. matched_literals_inverted (empty-background satisfaction) is
    # omitted so it doesn't flood the panel.
    return [(matched = ecs[j].matched_literals,
             failed = ecs[j].failed_literals .| ecs[j].failed_literals_inverted, vote = ecs[j].vote) for j in o[1:2]]
end
i_ok = idxs[1]
i_bad = findfirst(i -> predict(tm, xte[i]) != yte[i], eachindex(xte))
fig2 = Figure(size=(1700, 740))
for (row, i, tag) in ((1, i_ok, "correct"), (2, i_bad, "MISCLASSIFIED"))
    x = xte[i]; xb = tmbits(x); pc = predict(tm, x)
    occ = saliency(tm, x; occlude=:off)
    cf = counterfactual(tm, x, sortperm(occ, rev=true))
    rules = top_clause_rule(x, pc)
    a1 = Axis(fig2[row,1], aspect=DataAspect(), title="$tag: true $(yte[i]), pred $pc", titlesize=15); gray!(a1, xb)
    a2 = Axis(fig2[row,2], aspect=DataAspect(), title="signed saliency (IS/IS-NOT)", titlesize=14); signed!(a2, occ; underlay=xb)
    a3 = Axis(fig2[row,3], aspect=DataAspect(), title="counterfactual: $(length(cf.flips)) flips -> $(cf.new)", titlesize=14); cf_overlay!(a3, xb, cf.flips)
    a4 = Axis(fig2[row,4], aspect=DataAspect(), title="clause rule (vote $(rules[1].vote))", titlesize=14); clause_overlay!(a4, xb, rules[1].matched, rules[1].failed)
    a5 = Axis(fig2[row,5], aspect=DataAspect(), title="clause rule (vote $(rules[2].vote))", titlesize=14); clause_overlay!(a5, xb, rules[2].matched, rules[2].failed)
end
Label(fig2[0,:], "Per-instance explanation — overlays on the digit (green=clause satisfied, red=violated; cf: blue=removed, red=added)", fontsize=18)
CairoMakie.save(joinpath(OUT, "xai_instance.png"), fig2); println("saved xai_instance.png")

# ===================== figure 3: per-class saliency =====================
fig3 = Figure(size=(1000, 480))
for d in 0:9
    Xc = [xte[i] for i in eachindex(xte) if yte[i]==Int8(d)]
    smap,_ = class_saliency(tm, Xc, Int8(d); limit=40)
    ax = Axis(fig3[1+d÷5, d%5+1], aspect=DataAspect(), title=string(d)); support!(ax, smap); nodec!(ax)
end
Label(fig3[0,:], "Global per-class saliency (what supports each digit)", fontsize=24)
CairoMakie.save(joinpath(OUT, "xai_global.png"), fig3); println("saved xai_global.png")

# ===================== figure 4: global importance, necessity-by-violation, stability =====================
Xsub = xte[1:2000]
gimp, _ = global_importance(tm, Xsub; limit=300)
x0 = xte[i_ok]; xb0 = tmbits(x0); c0 = predict(tm, x0); cidx = top_clauses(tm, x0; k=1)[1].clause
ta0 = tm.clauses[findfirst(==(c0), tm.classes)]
incl = let l=@view(ta0.positive_included_literals[:,cidx]), li=@view(ta0.positive_included_literals_inverted[:,cidx])
    bv = BitVector(undef, ta0.clause_size); for i in eachindex(l); bv.chunks[i] = l[i]|li[i]; end
    Float64.([bv[i] for i in 1:ta0.clause_size]) end
Xclass = [xte[i] for i in eachindex(xte) if yte[i]==c0]
frac = clause_necessity(tm, c0, cidx, Xclass) ./ max(1, length(Xclass))
necessity = incl .* (1 .- frac)        # bright = included AND rarely violated = load-bearing
@printf("necessity: clause %d/class %d  %d included, mean load-bearing weight %.2f\n", cidx, c0, Int(sum(incl)), mean(necessity[incl.>0]))
xp = _copy(x0); for j in rand(rng, 1:length(x0), 8); xp[j] = !xp[j]; end
sal0 = saliency(tm, x0); salp = saliency(tm, xp); stab = cor(sal0, salp)
fig4 = Figure(size=(1700, 360))
g1=Axis(fig4[1,1],aspect=DataAspect(),title="global importance (all classes)",titlesize=14); heatmap!(g1, img28(gimp), colormap=:hot); nodec!(g1)
g2=Axis(fig4[1,2],aspect=DataAspect(),title="clause $cidx included literals",titlesize=14); support!(g2, incl; underlay=xb0)
g3=Axis(fig4[1,3],aspect=DataAspect(),title="necessity (bright = load-bearing)",titlesize=14); support!(g3, necessity; underlay=xb0)
g4=Axis(fig4[1,4],aspect=DataAspect(),title="saliency(x)",titlesize=14); support!(g4, max.(0.0,sal0))
g5=Axis(fig4[1,5],aspect=DataAspect(),title=@sprintf("stability diff (corr %.2f)", stab),titlesize=14); signed!(g5, sal0 .- salp; thresh=0.05)
Label(fig4[0,:], "Global importance · minimal-rule (necessity) · stability", fontsize=20)
CairoMakie.save(joinpath(OUT, "xai_global_necessity_stability.png"), fig4); println("saved xai_global_necessity_stability.png")
println("done.")
