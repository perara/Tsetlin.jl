# GLOBAL (and rule-level) XAI for TABULAR data with a Tsetlin Machine.
#
# Tabular features are encoded into bits (here: thermometer / quantile bins), so
# each feature is a GROUP of bits. Then the general XAI from src/utils/explain.jl
# applies directly, and `group_importance` aggregates bit-level attribution back
# to per-feature importance. Clauses translate to "feature > threshold" rules.
#
# Synthetic task (so we have ground truth): the class is fully determined by
# features 1 and 4 (a 2x2 quadrant -> 4 classes); features 2,3,5,6 are noise.
# A correct GLOBAL explanation must rank features 1 and 4 far above the rest.

include(joinpath(@__DIR__, "..", "src", "utils", "explain.jl"))
using .Tsetlin
using Random, Statistics, Printf

const NFEAT = 6
const N = 5000
rng = MersenneTwister(1)
Xraw = rand(rng, N, NFEAT)
y = Int8.([(Xraw[i,1] > 0.5) + 2 * (Xraw[i,4] > 0.5) for i in 1:N])   # classes 0..3

# thermometer thresholds per feature = quartiles (from data); each feature -> 3 bits
thresholds = [quantile(Xraw[:, f], [0.25, 0.5, 0.75]) for f in 1:NFEAT]
const NB = sum(length, thresholds)
feature_bits = let o = 0, fb = UnitRange{Int}[]
    for f in 1:NFEAT
        push!(fb, (o+1):(o+length(thresholds[f]))); o += length(thresholds[f])
    end
    fb
end

function encode(Xr)
    ins = Vector{TMInput}(undef, size(Xr, 1))
    for s in 1:size(Xr, 1)
        bv = falses(NB)
        for f in 1:NFEAT, k in eachindex(thresholds[f])
            bv[feature_bits[f][k]] = Xr[s, f] > thresholds[f][k]
        end
        ins[s] = TMInput(bv)
    end
    return ins
end

ntr = 4000
Xtr = encode(Xraw[1:ntr, :]); Ytr = y[1:ntr]
Xte = encode(Xraw[ntr+1:end, :]); Yte = y[ntr+1:end]
ys = Int8.(collect(0:3))
tm = TMClassifier(Xtr[1], ys, 40, 15, NB, 8, 1; states_num=256, include_limit=128)
train!(tm, Xtr, Ytr, Xte, Yte, 50; exclusive=true, verbose=0)
println("accuracy: ", round(accuracy(predict(tm, Xte), Yte) * 100, digits=1), "%\n")

# ---------- GLOBAL feature importance ----------
gimp, _ = global_importance(tm, Xte; limit=500)        # per-bit
fimp = group_importance(gimp, feature_bits)            # per-feature
fimp ./= maximum(fimp)
println("GLOBAL feature importance (relevant features are 1 and 4):")
for f in sortperm(fimp, rev=true)
    @printf("  feature %d : %5.2f %s\n", f, fimp[f], f in (1, 4) ? "  <-- truly relevant" : "")
end

# Shapley-based global importance (averaged over a sample), for comparison
sh = zeros(Float64, NB); m = 0
for i in 1:200; global m; sh .+= abs.(shapley(tm, Xte[i]; samples=40, rng=rng)); m += 1; end
fsh = group_importance(sh ./ m, feature_bits); fsh ./= maximum(fsh)
println("\nGLOBAL feature importance via Shapley:")
for f in sortperm(fsh, rev=true); @printf("  feature %d : %5.2f %s\n", f, fsh[f], f in (1,4) ? "  <--" : ""); end

# ---------- per-class feature importance ----------
println("\nper-class feature importance (top-2 features):")
for c in ys
    Xc = [Xte[i] for i in eachindex(Xte) if Yte[i] == c]
    sm, _ = class_saliency(tm, Xc, c; limit=200)
    println("  class $c : features ", sortperm(group_importance(sm, feature_bits), rev=true)[1:2])
end

# ---------- rules: translate a clause's literals into feature conditions ----------
bit_feature(b) = (for f in 1:NFEAT; b in feature_bits[f] && return (f, b - first(feature_bits[f]) + 1); end; (0, 0))
function clause_rule(cls, clause)
    ta = tm.clauses[findfirst(==(cls), tm.classes)]
    conds = String[]
    for (mask, op) in ((ta.positive_included_literals, ">"), (ta.positive_included_literals_inverted, "<="))
        col = @view mask[:, clause]
        bv = BitVector(undef, ta.clause_size); for i in eachindex(col); bv.chunks[i] = col[i]; end
        for b in 1:ta.clause_size
            bv[b] || continue
            (f, k) = bit_feature(b); f == 0 && continue
            push!(conds, @sprintf("f%d %s %.2f", f, op, thresholds[f][k]))
        end
    end
    return conds
end
println("\nexample rules (most-specific clause per class):")
for c in ys
    ta = tm.clauses[findfirst(==(c), tm.classes)]
    best = argmax([sum(count_ones, @view ta.positive_included_literals[:, j]) + sum(count_ones, @view ta.positive_included_literals_inverted[:, j]) for j in 1:size(ta.positive_included_literals, 2)])
    println("  class $c  IF  ", join(unique(clause_rule(c, best)), "  AND  "))
end
