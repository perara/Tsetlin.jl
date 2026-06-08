# GLOBAL XAI on a REAL tabular dataset (Iris) with a Tsetlin Machine.
# Each numeric feature is thermometer/quantile-encoded into bits (one group of
# bits per feature); then global_importance/shapley + group_importance give
# per-feature importance, and clauses translate to "feature > threshold" rules.
# Expectation (known): petal length & petal width dominate the species decision.

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
import Pkg
for p in ("MLDatasets", "DataFrames"); Base.find_package(p) === nothing && Pkg.add(p); end
include(joinpath(@__DIR__, "..", "src", "utils", "explain.jl"))
using .Tsetlin
using MLDatasets, DataFrames, Random, Statistics, Printf

d = MLDatasets.Iris(as_df=false)
fnames = d.metadata["feature_names"]            # sepallength, sepalwidth, petallength, petalwidth
Xall = d.features                               # 4 x 150 (feature x sample)
species = unique(vec(d.targets))
ymap = Dict(c => Int8(i - 1) for (i, c) in enumerate(species))
yall = Int8[ymap[t] for t in vec(d.targets)]
NFEAT = size(Xall, 1)

# thermometer encoding: each feature -> 3 bits at its quartiles
thresholds = [quantile(Xall[f, :], [0.25, 0.5, 0.75]) for f in 1:NFEAT]
const NB = sum(length, thresholds)
feature_bits = let o = 0, fb = UnitRange{Int}[]
    for f in 1:NFEAT; push!(fb, (o+1):(o+length(thresholds[f]))); o += length(thresholds[f]); end; fb
end
encode_col(col) = (bv = falses(NB); for f in 1:NFEAT, k in eachindex(thresholds[f]); bv[feature_bits[f][k]] = col[f] > thresholds[f][k]; end; TMInput(bv))

rng = MersenneTwister(1)
perm = randperm(rng, size(Xall, 2)); ntr = 120
tr = perm[1:ntr]; te = perm[ntr+1:end]
Xtr = [encode_col(Xall[:, s]) for s in tr]; Ytr = yall[tr]
Xte = [encode_col(Xall[:, s]) for s in te]; Yte = yall[te]
ys = sort(unique(yall))
tm = TMClassifier(Xtr[1], ys, 30, 12, NB, 8, 1; states_num=256, include_limit=128)
train!(tm, Xtr, Ytr, Xte, Yte, 200; exclusive=true, verbose=0)
@printf("Iris accuracy: %.1f%%\n\n", accuracy(predict(tm, Xte), Yte) * 100)

# ---------- GLOBAL feature importance ----------
gimp, _ = global_importance(tm, Xte)
fimp = group_importance(gimp, feature_bits); fimp ./= maximum(fimp)
println("GLOBAL feature importance (occlusion):")
for f in sortperm(fimp, rev=true); @printf("  %-12s : %.2f\n", fnames[f], fimp[f]); end

sh = zeros(NB); for x in Xte; sh .+= abs.(shapley(tm, x; samples=60, rng=rng)); end; sh ./= length(Xte)
fsh = group_importance(sh, feature_bits); fsh ./= maximum(fsh)
println("\nGLOBAL feature importance (Shapley):")
for f in sortperm(fsh, rev=true); @printf("  %-12s : %.2f\n", fnames[f], fsh[f]); end

# ---------- per-class top features ----------
println("\nper-class top features:")
for (i, c) in enumerate(ys)
    Xc = [Xte[j] for j in eachindex(Xte) if Yte[j] == c]; isempty(Xc) && continue
    ff = group_importance(class_saliency(tm, Xc, c)[1], feature_bits)
    println("  $(species[i]): ", join([fnames[f] for f in sortperm(ff, rev=true)[1:2]], ", "))
end

# ---------- rules ----------
bit_feature(b) = (for f in 1:NFEAT; b in feature_bits[f] && return (f, b - first(feature_bits[f]) + 1); end; (0, 0))
function clause_rule(cls, clause)
    ta = tm.clauses[findfirst(==(cls), tm.classes)]; conds = String[]
    for (mask, op) in ((ta.positive_included_literals, ">"), (ta.positive_included_literals_inverted, "<="))
        col = @view mask[:, clause]; bv = BitVector(undef, ta.clause_size); for i in eachindex(col); bv.chunks[i] = col[i]; end
        for b in 1:ta.clause_size; bv[b] || continue; (f, k) = bit_feature(b); f == 0 && continue; push!(conds, @sprintf("%s %s %.1f", fnames[f], op, thresholds[f][k])); end
    end
    conds
end
println("\nrule per species (most-specific clause):")
for (i, c) in enumerate(ys)
    ta = tm.clauses[findfirst(==(c), tm.classes)]
    best = argmax([sum(count_ones, @view ta.positive_included_literals[:, j]) + sum(count_ones, @view ta.positive_included_literals_inverted[:, j]) for j in 1:size(ta.positive_included_literals, 2)])
    println("  $(species[i])  IF  ", join(unique(clause_rule(c, best)), "  AND  "))
end
