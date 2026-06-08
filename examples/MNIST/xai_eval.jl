# Evaluate + visualize the XAI suite on MNIST:
#   saliency, exact clause decomposition, counterfactuals,
#   deletion/insertion faithfulness (saliency vs random baseline),
#   global importance, stability.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
import Pkg
for p in ("MLDatasets", "CairoMakie"); Base.find_package(p) === nothing && Pkg.add(p); end
using Statistics, Random, Printf
include(joinpath(@__DIR__, "..", "..", "src", "utils", "explain.jl"))
using .Tsetlin
using MLDatasets: MNIST
using CairoMakie

const EPOCHS = parse(Int, get(ENV, "XAI_EPOCHS", "40"))
const NEVAL  = parse(Int, get(ENV, "XAI_NEVAL", "25"))
const OUT = tempdir()
rng = MersenneTwister(1)

xtr, ytr = Tsetlin.unzip([MNIST(:train)...]); xte, yte = Tsetlin.unzip([MNIST(:test)...])
xtr = [Tsetlin.booleanize(x, 0.2) for x in xtr]; xte = [Tsetlin.booleanize(x, 0.2) for x in xte]
ytr = Int8.(ytr); yte = Int8.(yte)

tm = TMClassifier(xtr[1], ytr, 20, 20, 400, 150, 75, states_num=256, include_limit=128)
println("Training (epochs=$EPOCHS) ..."); train!(tm, xtr, ytr, xte, yte, EPOCHS; verbose=0)
println("accuracy: ", round(accuracy(predict(tm, xte), yte)*100, digits=2), "%")

img(v) = reverse(reshape(Float64.(v), 28, 28), dims=2)

# pick correctly-classified eval instances
idxs = Int[]; for i in randperm(rng, length(xte)); predict(tm, xte[i]) == yte[i] && (push!(idxs, i)); length(idxs) >= NEVAL && break; end

# ---------- faithfulness: saliency vs random ----------
steps = 25
del_s = zeros(steps+1); ins_s = zeros(steps+1); del_r = zeros(steps+1); ins_r = zeros(steps+1)
cf_s = Int[]; cf_r = Int[]; stab = Float64[]
for i in idxs
    x = xte[i]
    imp = saliency(tm, x)
    osal = sortperm(imp, rev=true)
    orand = randperm(rng, length(x))
    fs = faithfulness(tm, x, osal; steps=steps); fr = faithfulness(tm, x, orand; steps=steps)
    b = max(fs.base_margin, 1.0)
    del_s .+= fs.deletion ./ b; ins_s .+= fs.insertion ./ b
    del_r .+= fr.deletion ./ b; ins_r .+= fr.insertion ./ b
    push!(cf_s, (c = counterfactual(tm, x, osal); c.success ? length(c.flips) : length(x)))
    push!(cf_r, (c = counterfactual(tm, x, orand); c.success ? length(c.flips) : length(x)))
    # stability: flip 5 random bits, recompute saliency, correlate
    x2 = _copy(x); for j in rand(rng, 1:length(x), 5); x2[j] = !x2[j]; end
    push!(stab, cor(imp, saliency(tm, x2)))
end
del_s ./= length(idxs); ins_s ./= length(idxs); del_r ./= length(idxs); ins_r ./= length(idxs)
auc(v) = sum(v)/length(v)

println("\n=== Faithfulness (normalized margin, avg over $(length(idxs)) instances) ===")
@printf("deletion  AUC: saliency %.3f | random %.3f   (lower = better)\n", auc(del_s), auc(del_r))
@printf("insertion AUC: saliency %.3f | random %.3f   (higher = better)\n", auc(ins_s), auc(ins_r))
@printf("=== Counterfactual: avg flips to change class: saliency %.1f | random %.1f ===\n", mean(cf_s), mean(cf_r))
@printf("=== Stability: saliency corr under 5-bit perturbation: %.3f ===\n", mean(stab))

# clause decomposition exactness check
xx = xte[idxs[1]]
tc = top_clauses(tm, xx; k=3)
allpos = top_clauses(tm, xx; k=10^9)
println("=== Clause decomposition: exact? sum(clause votes) reproduces class pos-score: ",
        sum(c.vote for c in allpos), " (top clause votes: ", [c.vote for c in tc], ") ===")

# ---------- figure 1: faithfulness curves ----------
set_theme!(theme_black())
fig1 = Figure(size=(1100, 460))
ax1 = Axis(fig1[1,1], title="Deletion (faithful = drops fast)", xlabel="fraction removed", ylabel="norm. margin")
lines!(ax1, range(0,1;length=steps+1), del_s, label="saliency", linewidth=3)
lines!(ax1, range(0,1;length=steps+1), del_r, label="random", linewidth=3, linestyle=:dash); axislegend(ax1)
ax2 = Axis(fig1[1,2], title="Insertion (faithful = rises fast)", xlabel="fraction inserted", ylabel="norm. margin")
lines!(ax2, range(0,1;length=steps+1), ins_s, label="saliency", linewidth=3)
lines!(ax2, range(0,1;length=steps+1), ins_r, label="random", linewidth=3, linestyle=:dash); axislegend(ax2, position=:rb)
Label(fig1[0,:], "Faithfulness: saliency vs random", fontsize=24)
p1 = joinpath(OUT, "xai_faithfulness.png"); CairoMakie.save(p1, fig1); println("saved: $p1")

# ---------- figure 2: instance explanation ----------
i0 = idxs[1]; x0 = xte[i0]
imp0 = saliency(tm, x0); osal0 = sortperm(imp0, rev=true)
cf0 = counterfactual(tm, x0, osal0)
flipmap = zeros(Float64, length(x0)); for f in cf0.flips; flipmap[f] = 1.0; end
tcs = top_clauses(tm, x0; k=3)
fig2 = Figure(size=(1700, 360))
panels = [("input (true $(yte[i0]), pred $(predict(tm,x0)))", Float64.([x0[i] for i in 1:length(x0)]), :hot),
          ("saliency", max.(0.0, imp0), :hot),
          ("counterfactual flips ($(length(cf0.flips))) -> $(cf0.new)", flipmap, :viridis),
          ("clause $(tcs[1].clause) (vote $(tcs[1].vote))", Float64.(tcs[1].matched), :hot),
          ("clause $(tcs[2].clause) (vote $(tcs[2].vote))", Float64.(tcs[2].matched), :hot),
          ("clause $(tcs[3].clause) (vote $(tcs[3].vote))", Float64.(tcs[3].matched), :hot)]
for (j,(t,v,cm)) in enumerate(panels)
    ax = Axis(fig2[1,j], aspect=DataAspect(), title=t, titlesize=14); heatmap!(ax, img(v), colormap=cm); hidedecorations!(ax); hidespines!(ax)
end
Label(fig2[0,:], "Per-instance explanation: saliency, counterfactual, exact clause reasons", fontsize=22)
p2 = joinpath(OUT, "xai_instance.png"); CairoMakie.save(p2, fig2); println("saved: $p2")

# ---------- figure 3: global importance (avg saliency per class) ----------
fig3 = Figure(size=(1000, 480))
for d in 0:9
    Xc = [xte[i] for i in eachindex(xte) if yte[i]==Int8(d)]
    smap,_ = class_saliency(tm, Xc, Int8(d); limit=30)
    ax = Axis(fig3[1+d÷5, d%5+1], aspect=DataAspect(), title=string(d)); heatmap!(ax, img(max.(0.0,smap)), colormap=:hot); hidedecorations!(ax); hidespines!(ax)
end
Label(fig3[0,:], "Global per-class saliency", fontsize=24)
p3 = joinpath(OUT, "xai_global.png"); CairoMakie.save(p3, fig3); println("saved: $p3")

# ---------- figure 4: global importance, minimal-rule (necessity), stability ----------
Xsub = xte[1:2000]
gimp, _ = global_importance(tm, Xsub; limit=300)

# minimal-rule reduction of the instance's top clause
c0 = predict(tm, x0); cidx = top_clauses(tm, x0; k=1)[1].clause
ta0 = tm.clauses[findfirst(==(c0), tm.classes)]
incl = let l = @view(ta0.positive_included_literals[:, cidx]), li = @view(ta0.positive_included_literals_inverted[:, cidx])
    bv = BitVector(undef, ta0.clause_size); for i in eachindex(l); bv.chunks[i] = l[i] | li[i]; end
    Float64.([bv[i] for i in 1:ta0.clause_size])
end
# Template extraction: count violations on the clause's OWN class; literals
# consistently satisfied there are the load-bearing template, the rest is padding
# the LF margin tolerates.
Xclass = [xte[i] for i in eachindex(xte) if yte[i] == c0]
viol = clause_necessity(tm, c0, cidx, Xclass)
frac = viol ./ max(1, length(Xclass))
loadbearing = incl .* (frac .< 0.15); redundant = incl .* (frac .>= 0.15)
@printf("=== Minimal rule (clause %d, class %d): %d included -> %d load-bearing template, %d padding ===\n",
        cidx, c0, Int(sum(incl)), Int(sum(loadbearing .> 0)), Int(sum(redundant .> 0)))

# stability view: saliency on x0 vs a 8-bit-perturbed copy
xp = _copy(x0); for j in rand(rng, 1:length(x0), 8); xp[j] = !xp[j]; end
sal0 = max.(0.0, saliency(tm, x0)); salp = max.(0.0, saliency(tm, xp))

fig4 = Figure(size=(1700, 360))
g4 = [("global importance (all classes)", gimp, :hot),
      ("clause $cidx: included literals", incl, :hot),
      ("load-bearing template", loadbearing, :hot),
      ("padding (LF-tolerated)", redundant, :hot),
      ("saliency (x)", sal0, :hot),
      ("saliency (x perturbed 8 bits)", salp, :hot)]
for (j,(t,v,cm)) in enumerate(g4)
    ax = Axis(fig4[1,j], aspect=DataAspect(), title=t, titlesize=14); heatmap!(ax, img(v), colormap=cm); hidedecorations!(ax); hidespines!(ax)
end
Label(fig4[0,:], "Global importance, minimal-rule (necessity), stability", fontsize=22)
p4 = joinpath(OUT, "xai_global_necessity_stability.png"); CairoMakie.save(p4, fig4); println("saved: $p4")
