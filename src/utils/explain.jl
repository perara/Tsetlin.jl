include("../Tsetlin.jl")

export explain, saliency, class_saliency

using Base.Threads
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

`occlude`:
  `:off`  set bit 1->0 (classic remove-evidence; already-0 bits get 0 and are skipped)
  `:on`   set bit 0->1
  `:flip` flip every bit (use for dense/balanced inputs where there is no "off")
"""
function saliency(tm::TMClassifier{ClassType}, x::TMInput; target::Union{ClassType, Nothing}=nothing, occlude::Symbol=:off, index::Bool=false)::Vector{Float64} where ClassType
    occlude in (:off, :on, :flip) || error("occlude must be :off, :on, or :flip")
    K = length(tm.classes)
    s = Vector{Int64}(undef, K)
    s2 = Vector{Int64}(undef, K)
    _class_scores!(s, tm, x; index=index)
    ti = target === nothing ? argmax(s) : findfirst(==(target), tm.classes)
    ti === nothing && error("target $target is not one of tm.classes")
    base = _margin(s, ti)
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
        imp[i] = base - _margin(s2, ti)
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
