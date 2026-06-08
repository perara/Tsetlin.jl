include("../Tsetlin.jl")

export explain, saliency, class_saliency, top_clauses, faithfulness, counterfactual, global_importance, clause_necessity, shapley, group_importance

using Base.Threads
using Random: shuffle!, default_rng
using .Tsetlin: TMInput, TMClassifier, TATeam, predict


struct ExplainedLiteralSum
    positive_included_literals::Vector{Int16}
    positive_included_literals_inverted::Vector{Int16}
    negative_included_literals::Vector{Int16}
    negative_included_literals_inverted::Vector{Int16}
end


struct ExplainedClause
    matched_literals::BitVector
    matched_literals_inverted::BitVector
    failed_literals::BitVector
    failed_literals_inverted::BitVector
    vote::Int64
end


struct ExplainedClauses
    clauses::Vector{ExplainedClause}
    votes::Int64
end


@inline function explain(literals::Matrix{UInt64}, clause_size::Int64)::Vector{Int16}
    res = zeros(Int16, clause_size)
    bv = BitVector(undef, clause_size)
    @inbounds for lits in eachcol(literals)
        copyto!(bv.chunks, lits)
        @simd for i in 1:clause_size
            res[i] += bv[i]
        end
    end
    return res
end


@inline function explain(ta::TATeam)::ExplainedLiteralSum
    return ExplainedLiteralSum(
        explain(ta.positive_included_literals, ta.clause_size),
        explain(ta.positive_included_literals_inverted, ta.clause_size),
        explain(ta.negative_included_literals, ta.clause_size),
        explain(ta.negative_included_literals_inverted, ta.clause_size),
    )
end


function explain(tm::TMClassifier{ClassType})::ExplainedLiteralSum where ClassType <: Bool
    return explain(tm.clauses)
end


function explain(tm::TMClassifier{ClassType})::Dict{ClassType, ExplainedLiteralSum} where ClassType
    res::Dict{ClassType, ExplainedLiteralSum} = Dict()
    @inbounds for (cls, ta) in zip(tm.classes, tm.clauses)
        res[cls] = explain(ta)
    end
    return res
end


function explain(tm::TMClassifier{<:Any, N}, x::TMInput, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64})::ExplainedClause where N
    matched_literals = BitVector(undef, x.len)
    matched_literals_inverted = BitVector(undef, x.len)
    failed_literals = BitVector(undef, x.len)
    failed_literals_inverted = BitVector(undef, x.len)
    c = 0
    @inbounds for i in 1:N
        matched_literals.chunks[i] = x.chunks[i] & literals[i]
        matched_literals_inverted.chunks[i] = ~x.chunks[i] & literals_inverted[i]
        failed_literals.chunks[i] = ~x.chunks[i] & literals[i]
        failed_literals_inverted.chunks[i] = x.chunks[i] & literals_inverted[i]
        c += count_ones(failed_literals.chunks[i] | failed_literals_inverted.chunks[i])
    end
    return ExplainedClause(
        matched_literals,
        matched_literals_inverted,
        failed_literals,
        failed_literals_inverted,
        max(0, tm.LF - c),
    )
end


function explain(tm::TMClassifier{<:Any, <:Any, <:Any, <:Any, C}, ta::TATeam, x::TMInput)::Tuple{ExplainedClauses, ExplainedClauses} where C
    pos = Vector{ExplainedClause}(undef, C)
    neg = Vector{ExplainedClause}(undef, C)
    @inbounds for i in 1:C
        pos[i] = explain(tm, x, @view(ta.positive_included_literals[:, i]), @view(ta.positive_included_literals_inverted[:, i]))
        neg[i] = explain(tm, x, @view(ta.negative_included_literals[:, i]), @view(ta.negative_included_literals_inverted[:, i]))
    end
    return (
        ExplainedClauses(pos, sum(c.vote for c in pos)),
        ExplainedClauses(neg, sum(c.vote for c in neg)),
    )
end


function explain(tm::TMClassifier{ClassType}, x::TMInput)::Dict{ClassType, ExplainedClauses} where ClassType <: Bool
    pos, neg = explain(tm, tm.clauses, x)
    return Dict(
        true => pos,
        false => neg,
    )
end


function explain(tm::TMClassifier{ClassType}, x::TMInput)::Dict{ClassType, Dict{Bool, ExplainedClauses}} where ClassType
    res::Dict{ClassType, Dict{Bool, ExplainedClauses}} = Dict()
    @inbounds for (cls, ta) in zip(tm.classes, tm.clauses)
        pos, neg = explain(tm, ta, x)
        res[cls] = Dict(
            true => pos,
            false => neg,
        )
    end
    return res
end


function explain(tm::TMClassifier{ClassType}, X::Vector{TMInput})::Vector{Dict{ClassType, Dict{Bool, ExplainedClauses}}} where ClassType
    res = Vector{Dict{ClassType, Dict{Bool, ExplainedClauses}}}(undef, length(X))
    @threads for i in eachindex(X)
        res[i] = explain(tm, X[i])
    end
    return res
end


# ---------------------------------------------------------------------------
# Occlusion saliency: a domain-agnostic, model-faithful attribution that works
# on any TMClassifier (binary or multiclass) and any input size. It measures
# how much each input bit contributes to the predicted/target class decision by
# how far the class margin drops when that bit is occluded. Returns a plain
# Vector{Float64} over input bits (length == length(x)); reshape it to your
# modality (e.g. 28x28 for images) only when visualising.
# ---------------------------------------------------------------------------

# Per-class scores (positive vote - negative vote), aligned with tm.classes.
@inline function _class_scores!(s::Vector{Int64}, tm::TMClassifier{<:Bool}, x::TMInput; index::Bool=false)
    p, n = Tsetlin.vote(tm, tm.clauses, x; index=index)
    s[1] = p - n   # tm.classes[1] == true
    s[2] = n - p   # tm.classes[2] == false
    return s
end

@inline function _class_scores!(s::Vector{Int64}, tm::TMClassifier, x::TMInput; index::Bool=false)
    @inbounds for k in eachindex(tm.clauses)
        p, n = Tsetlin.vote(tm, tm.clauses[k], x; index=index)
        s[k] = p - n
    end
    return s
end

# Margin of class index ti = its score minus the best competing class score.
@inline function _margin(s::Vector{Int64}, ti::Int)::Int64
    best = typemin(Int64)
    @inbounds for j in eachindex(s)
        j == ti && continue
        s[j] > best && (best = s[j])
    end
    return s[ti] - best
end

"""
    saliency(tm, x; target=nothing, occlude=:off, index=false) -> Vector{Float64}

Per-bit occlusion saliency for a single input `x`. `importance[i]` is the drop
in the `target` class margin when bit `i` is occluded (so positive = the bit
supports the target class). `target` defaults to the predicted class.

The returned values are SIGNED: positive = the bit supports the target class,
negative = the bit is evidence AGAINST it (toward another class). Render with a
diverging colormap to see "what it is" (red) vs "what it is not" (blue).

`versus`: if given, contrast against that specific class instead of the best
competitor -- i.e. "why `target` and not `versus`" (positive favours `target`,
negative favours `versus`).

`occlude`:
  `:off`  set bit 1->0 (classic remove-evidence; already-0 bits get 0 and are skipped)
  `:on`   set bit 0->1
  `:flip` flip every bit (use for dense/balanced inputs where there is no "off")
"""
function saliency(tm::TMClassifier{ClassType}, x::TMInput; target::Union{ClassType, Nothing}=nothing, versus::Union{ClassType, Nothing}=nothing, occlude::Symbol=:off, index::Bool=false)::Vector{Float64} where ClassType
    occlude in (:off, :on, :flip) || error("occlude must be :off, :on, or :flip")
    K = length(tm.classes)
    s = Vector{Int64}(undef, K)
    s2 = Vector{Int64}(undef, K)
    _class_scores!(s, tm, x; index=index)
    ti = target === nothing ? argmax(s) : findfirst(==(target), tm.classes)
    ti === nothing && error("target $target is not one of tm.classes")
    vi = 0
    if versus !== nothing
        vi = something(findfirst(==(versus), tm.classes), 0)
        vi == 0 && error("versus $versus is not one of tm.classes")
    end
    contrast(sv) = vi == 0 ? _margin(sv, ti) : Int64(sv[ti] - sv[vi])
    base = contrast(s)
    imp = zeros(Float64, length(x))
    @inbounds for i in 1:length(x)
        b = x[i]
        if occlude === :off
            b || continue
            nv = false
        elseif occlude === :on
            b && continue
            nv = true
        else
            nv = !b
        end
        x[i] = nv
        _class_scores!(s2, tm, x; index=index)
        x[i] = b                      # restore
        imp[i] = base - contrast(s2)
    end
    return imp
end

"""
    class_saliency(tm, X, target; occlude=:off, index=false, only_correct=true, limit=typemax(Int))

Average `saliency` over the inputs in `X` for class `target`, giving a
class-level attribution map. By default only inputs the model classifies as
`target` are included. Returns `(map::Vector{Float64}, n_used::Int)`.
"""
function class_saliency(tm::TMClassifier{ClassType}, X::AbstractVector{TMInput}, target::ClassType; occlude::Symbol=:off, index::Bool=false, only_correct::Bool=true, limit::Int=typemax(Int)) where ClassType
    isempty(X) && error("X is empty")
    acc = zeros(Float64, length(first(X)))
    cnt = 0
    for x in X
        cnt >= limit && break
        if only_correct && predict(tm, x; index=index) != target
            continue
        end
        acc .+= saliency(tm, x; target=target, occlude=occlude, index=index)
        cnt += 1
    end
    cnt > 0 && (acc ./= cnt)
    return acc, cnt
end


_copy(x::TMInput) = (c = Memory{UInt64}(undef, length(x.chunks)); copyto!(c, x.chunks); TMInput(c, x.len))

"""
    group_importance(v, groups; agg = g -> sum(abs, g)) -> Vector

Aggregate a per-bit attribution `v` (from `global_importance`, `saliency`,
`shapley`, ...) to per-group values. `groups[i]` is the collection of bit indices
belonging to group `i`. For tabular data each original feature is one group of
encoding bits, so this turns bit-level attribution into feature-level importance.
Default sums magnitudes; pass `agg = sum` for a signed total.
"""
group_importance(v::AbstractVector, groups; agg = g -> sum(abs, g)) = [agg(v[g]) for g in groups]

"""
    shapley(tm, x; target=nothing, versus=nothing, samples=50, baseline=nothing, index=false, rng=default_rng())

Monte-Carlo (permutation-sampled) Shapley values for the `target` class margin.
Unlike single-bit occlusion, Shapley credits each bit by its *average marginal*
contribution across random coalitions, so it shares credit fairly between
interacting/redundant bits. Signed (positive = supports `target`, negative =
against), supports `versus` for contrast, and satisfies the efficiency property
`sum(phi) ≈ margin(x) - margin(baseline)`.

Only bits where `x` differs from `baseline` (default all-zero) get nonzero value,
so the cost is `samples * (#differing bits) * predict`.
"""
function shapley(tm::TMClassifier{ClassType}, x::TMInput; target::Union{ClassType, Nothing}=nothing, versus::Union{ClassType, Nothing}=nothing, samples::Int=50, baseline::Union{TMInput, Nothing}=nothing, index::Bool=false, rng=default_rng())::Vector{Float64} where ClassType
    n = length(x)
    base_in = baseline === nothing ? TMInput(n) : baseline
    K = length(tm.classes)
    s = Vector{Int64}(undef, K)
    _class_scores!(s, tm, x; index=index)
    ti = target === nothing ? argmax(s) : findfirst(==(target), tm.classes)
    ti === nothing && error("target $target is not one of tm.classes")
    vi = 0
    if versus !== nothing
        vi = something(findfirst(==(versus), tm.classes), 0)
        vi == 0 && error("versus $versus is not one of tm.classes")
    end
    contrast(sv) = vi == 0 ? _margin(sv, ti) : Int64(sv[ti] - sv[vi])

    active = Int[]
    @inbounds for i in 1:n
        x[i] != base_in[i] && push!(active, i)
    end
    phi = zeros(Float64, n)
    xc = _copy(base_in)
    for _ in 1:samples
        shuffle!(rng, active)
        _class_scores!(s, tm, xc; index=index)
        prev = contrast(s)
        @inbounds for i in active
            xc[i] = x[i]
            _class_scores!(s, tm, xc; index=index)
            cur = contrast(s)
            phi[i] += cur - prev
            prev = cur
        end
        @inbounds for i in active
            xc[i] = base_in[i]
        end
    end
    phi ./= samples
    return phi
end

"""
    top_clauses(tm, x; target=nothing, k=5, index=false)

Exact clause-level decomposition of a prediction: the `k` highest-voting clauses
for the `target` class (default: the predicted class), each as a NamedTuple with
its `clause` index, `vote`, and the `matched`/`matched_inverted` literal bitmasks
(the conjunction that actually fired). This is the model's real computation, not
an approximation, so it is maximally faithful.
"""
function top_clauses(tm::TMClassifier{ClassType}, x::TMInput; target::Union{ClassType, Nothing}=nothing, k::Int=5, index::Bool=false) where ClassType
    cls = target === nothing ? predict(tm, x; index=index) : target
    ta = ClassType <: Bool ? tm.clauses : tm.clauses[findfirst(==(cls), tm.classes)]
    C = size(ta.positive_included_literals, 2)
    ecs = [explain(tm, x, @view(ta.positive_included_literals[:, i]), @view(ta.positive_included_literals_inverted[:, i])) for i in 1:C]
    ord = sortperm(ecs; by = c -> c.vote, rev=true)
    return [(clause = i, vote = ecs[i].vote, matched = ecs[i].matched_literals, matched_inverted = ecs[i].matched_literals_inverted) for i in ord[1:min(k, C)]]
end

# Margin (target score minus best competitor) for a given input, reusing buffer s.
@inline function _margin_at(tm, x::TMInput, ti::Int, s::Vector{Int64}; index::Bool=false)
    _class_scores!(s, tm, x; index=index)
    return _margin(s, ti)
end

"""
    faithfulness(tm, x, order; target=nothing, index=false, steps=25)

Deletion/insertion faithfulness test for an attribution `order` (bit indices,
most-important first). Deletion progressively sets the top bits to 0 (margin
should drop fast for a faithful order); insertion progressively reveals them on
a blank input (margin should rise fast). Returns a NamedTuple with the curves,
their `fracs`, and AUCs (`deletion_auc` lower = better, `insertion_auc` higher =
better). Domain-agnostic.
"""
function faithfulness(tm::TMClassifier{ClassType}, x::TMInput, order::AbstractVector{<:Integer}; target::Union{ClassType, Nothing}=nothing, index::Bool=false, steps::Int=25) where ClassType
    K = length(tm.classes)
    s = Vector{Int64}(undef, K)
    _class_scores!(s, tm, x; index=index)
    ti = target === nothing ? argmax(s) : findfirst(==(target), tm.classes)
    n = length(x)
    fracs = collect(range(0.0, 1.0; length=steps + 1))

    del = Float64[]; xd = _copy(x); prev = 0
    for f in fracs
        kk = round(Int, f * n)
        @inbounds for j in (prev + 1):kk; xd[order[j]] = false; end
        prev = kk
        push!(del, _margin_at(tm, xd, ti, s; index=index))
    end

    ins = Float64[]; xi = TMInput(n); prev = 0
    for f in fracs
        kk = round(Int, f * n)
        @inbounds for j in (prev + 1):kk; xi[order[j]] = x[order[j]]; end
        prev = kk
        push!(ins, _margin_at(tm, xi, ti, s; index=index))
    end

    return (; fracs, deletion = del, insertion = ins,
            deletion_auc = sum(del) / length(del), insertion_auc = sum(ins) / length(ins),
            base_margin = del[1])
end

"""
    counterfactual(tm, x, order; index=false, max_flips=length(x))

Greedy counterfactual: flip bits in `order` (e.g. saliency-descending) one at a
time until the prediction changes. Returns `(flips, original, new, success)` —
the minimal-by-this-order set of bit flips that changes the decision.
"""
function counterfactual(tm::TMClassifier, x::TMInput, order::AbstractVector{<:Integer}; index::Bool=false, max_flips::Int=length(x))
    orig = predict(tm, x; index=index)
    xc = _copy(x); flips = Int[]
    @inbounds for j in 1:min(max_flips, length(order))
        i = order[j]; xc[i] = !xc[i]; push!(flips, i)
        if predict(tm, xc; index=index) != orig
            return (; flips, original = orig, new = predict(tm, xc; index=index), success = true)
        end
    end
    return (; flips, original = orig, new = predict(tm, xc; index=index), success = false)
end

"""
    global_importance(tm, X; occlude=:off, index=false, limit=typemax(Int))

Dataset-level feature importance: the mean absolute per-bit saliency over the
inputs in `X` (each explained for its own predicted class). Returns
`(map::Vector{Float64}, n_used::Int)` over input bits -- which features the
model relies on overall, independent of class.
"""
function global_importance(tm::TMClassifier, X::AbstractVector{TMInput}; occlude::Symbol=:off, index::Bool=false, limit::Int=typemax(Int))
    isempty(X) && error("X is empty")
    acc = zeros(Float64, length(first(X)))
    cnt = 0
    for x in X
        cnt >= limit && break
        acc .+= abs.(saliency(tm, x; occlude=occlude, index=index))
        cnt += 1
    end
    cnt > 0 && (acc ./= cnt)
    return acc, cnt
end

"""
    clause_necessity(tm, target, clause, X; index=false) -> Vector{Int}

Per-literal necessity for one clause of class `target`: how often each literal
position is *violated* across the inputs `X`. In the graded clause a satisfied
literal never changes the vote, so a literal violated zero times over `X` is
redundant padding and can be dropped without changing the clause's behaviour on
that data. Use it to strip a fuzzy clause to its load-bearing literals.
"""
function clause_necessity(tm::TMClassifier{ClassType}, target::ClassType, clause::Int, X::AbstractVector{TMInput}; index::Bool=false) where ClassType
    ta = ClassType <: Bool ? tm.clauses : tm.clauses[findfirst(==(target), tm.classes)]
    l = @view(ta.positive_included_literals[:, clause])
    li = @view(ta.positive_included_literals_inverted[:, clause])
    n = ta.clause_size
    nch = length(l)
    counts = zeros(Int, n)
    failed = BitVector(undef, n)
    @inbounds for x in X
        for i in 1:nch
            failed.chunks[i] = (~x.chunks[i] & l[i]) | (x.chunks[i] & li[i])
        end
        for i in 1:n
            counts[i] += failed[i]
        end
    end
    return counts
end
