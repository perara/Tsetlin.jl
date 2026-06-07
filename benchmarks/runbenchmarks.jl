# Dependency-free benchmark runner for Tsetlin.jl.
#
# Usage:
#   julia -O3 -t auto benchmarks/runbenchmarks.jl --quick
#   julia -O3 -t auto benchmarks/runbenchmarks.jl --mode=exhaustive --format=markdown --output=benchmarks/results.md

using Dates
using Printf
using Random
using Serialization
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(REPO_ROOT, "src", "utils", "explain.jl"))
include(joinpath(REPO_ROOT, "src", "utils", "fastconv.jl"))
include(joinpath(REPO_ROOT, "examples", "TEXT", "config.jl"))
include(joinpath(REPO_ROOT, "examples", "TEXT", "HDC.jl"))

const TS = Tsetlin

Base.@kwdef mutable struct BenchOptions
    mode::Symbol = :quick
    repeats::Int = 5
    min_sample_ns::UInt64 = UInt64(10_000_000)
    max_loops::Int = 1_000_000
    seed::Int = 1
    format::Symbol = :markdown
    output::String = ""
    gc_each_sample::Bool = true
    include_io::Bool = true
    include_hdc::Bool = true
    include_builtin_benchmark::Bool = false
end

struct BenchCase
    group::String
    name::String
    make::Function
    run::Function
    unit::String
    note::String
end

struct BenchResult
    group::String
    name::String
    unit::String
    loops::Int
    repeats::Int
    median_ns::Float64
    min_ns::Float64
    max_ns::Float64
    allocated_bytes::Int
    note::String
end

function usage()
    println("""
    Tsetlin.jl benchmark suite

    Options:
      --quick                    Small, fast smoke benchmark suite.
      --full                     Larger benchmark suite for local comparisons.
      --exhaustive               Broad grid over sizes, densities, and modes.
      --mode=quick|full|exhaustive
      --repeats=N                Samples per case after warmup.
      --min-sample-ms=N          Autotune each case to at least N ms per sample.
      --max-loops=N              Maximum inner loops during autotuning.
      --seed=N                   Random seed.
      --format=markdown|csv|json Output format.
      --output=PATH              Write output to PATH instead of stdout.
      --no-io                    Skip save/load benchmarks.
      --no-hdc                   Skip TEXT HDC benchmarks.
      --include-built-in         Include the existing TS.benchmark smoke case.
      --no-gc                    Do not run GC before each timed sample.
      --help                     Show this message.
    """)
end

function parse_args(args)::BenchOptions
    opts = BenchOptions()
    mode_set = false
    for arg in args
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--quick"
            opts.mode = :quick
            mode_set = true
        elseif arg == "--full"
            opts.mode = :full
            mode_set = true
        elseif arg == "--exhaustive"
            opts.mode = :exhaustive
            mode_set = true
        elseif startswith(arg, "--mode=")
            opts.mode = Symbol(split(arg, "=", limit=2)[2])
            mode_set = true
        elseif startswith(arg, "--repeats=")
            opts.repeats = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--min-sample-ms=")
            ms = parse(Float64, split(arg, "=", limit=2)[2])
            opts.min_sample_ns = UInt64(round(Int, ms * 1_000_000))
        elseif startswith(arg, "--max-loops=")
            opts.max_loops = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            opts.seed = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--format=")
            opts.format = Symbol(lowercase(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--output=")
            opts.output = split(arg, "=", limit=2)[2]
        elseif arg == "--no-io"
            opts.include_io = false
        elseif arg == "--no-hdc"
            opts.include_hdc = false
        elseif arg == "--include-built-in"
            opts.include_builtin_benchmark = true
        elseif arg == "--no-gc"
            opts.gc_each_sample = false
        else
            error("Unknown benchmark option: $arg")
        end
    end

    opts.mode in (:quick, :full, :exhaustive) || error("--mode must be quick, full, or exhaustive")
    opts.format in (:markdown, :csv, :json) || error("--format must be markdown, csv, or json")
    opts.repeats > 0 || error("--repeats must be positive")
    opts.max_loops > 0 || error("--max-loops must be positive")

    if !mode_set
        opts.mode = :quick
    end
    if opts.mode == :full && opts.repeats == 5
        opts.repeats = 7
        opts.min_sample_ns = UInt64(25_000_000)
    elseif opts.mode == :exhaustive && opts.repeats == 5
        opts.repeats = 9
        opts.min_sample_ns = UInt64(50_000_000)
    end
    return opts
end

# Opaque, non-deletable call boundary. Without this the optimizer inlines the
# (specialized) case body into the timing loop and, for pure read-only cases,
# either deletes the unused result (DCE) or computes it once and hoists it out
# of the loop (LICM) -- such cases then measure as 0.0 ns. `@noinline` keeps the
# body out of the loop and `Base.donotdelete` marks the result as observed, so
# the work is forced to run every iteration. It stays specialized on F, so there
# is no boxing or dynamic-dispatch overhead.
@noinline function timed_call(f::F, state) where {F}
    Base.donotdelete(f(state))
    return nothing
end

function elapsed_ns(f::Function, state, loops::Int)::UInt64
    start = time_ns()
    @inbounds for _ in 1:loops
        timed_call(f, state)
    end
    return time_ns() - start
end

function autotune(case::BenchCase, opts::BenchOptions)::Int
    state = case.make()
    loops = 1
    while loops < opts.max_loops
        ns = elapsed_ns(case.run, state, loops)
        ns >= opts.min_sample_ns && return loops
        loops = min(loops * 2, opts.max_loops)
    end
    return loops
end

function allocated_bytes(case::BenchCase)::Int
    state = case.make()
    return @allocated timed_call(case.run, state)
end

function quiet(f::Function)
    redirect_stdout(devnull) do
        f()
    end
end

function run_case(case::BenchCase, opts::BenchOptions)::BenchResult
    for _ in 1:2
        state = case.make()
        case.run(state)
    end

    loops = autotune(case, opts)
    samples = Vector{Float64}(undef, opts.repeats)
    for i in 1:opts.repeats
        opts.gc_each_sample && GC.gc()
        state = case.make()
        ns = elapsed_ns(case.run, state, loops)
        samples[i] = Float64(ns) / loops
    end
    return BenchResult(
        case.group,
        case.name,
        case.unit,
        loops,
        opts.repeats,
        median(samples),
        minimum(samples),
        maximum(samples),
        allocated_bytes(case),
        case.note,
    )
end

function run_cases(cases::Vector{BenchCase}, opts::BenchOptions)::Vector{BenchResult}
    results = BenchResult[]
    total = length(cases)
    for (i, case) in enumerate(cases)
        @printf("[%3d/%3d] %-18s %s\n", i, total, case.group, case.name)
        flush(stdout)
        push!(results, run_case(case, opts))
    end
    return results
end

function rand_bitvector(n::Int, density::Float64)::BitVector
    bv = falses(n)
    @inbounds for i in eachindex(bv)
        bv[i] = rand() < density
    end
    return bv
end

rand_input(n::Int, density::Float64) = TS.TMInput(rand_bitvector(n, density))

function set_masks!(literals::AbstractMatrix{UInt64}, clause_size::Int, density::Float64)
    fill!(literals, zero(UInt64))
    @inbounds for col in axes(literals, 2)
        for bit in 1:clause_size
            if rand() < density
                chunk = ((bit - 1) >>> 6) + 1
                shift = (bit - 1) & 63
                literals[chunk, col] |= one(UInt64) << shift
            end
        end
    end
    return literals
end

function randomize_masks!(ta::TS.TATeam; density::Float64=0.05, contradictions::Bool=true)
    set_masks!(ta.positive_included_literals, ta.clause_size, density)
    set_masks!(ta.negative_included_literals, ta.clause_size, density)
    if contradictions
        set_masks!(ta.positive_included_literals_inverted, ta.clause_size, density)
        set_masks!(ta.negative_included_literals_inverted, ta.clause_size, density)
    else
        fill!(ta.positive_included_literals_inverted, zero(UInt64))
        fill!(ta.negative_included_literals_inverted, zero(UInt64))
    end
    return ta
end

function randomize_masks!(tm::TS.TMClassifier; density::Float64=0.05, contradictions::Bool=true)
    if typeof(first(tm.classes)) == Bool
        randomize_masks!(tm.clauses; density=density, contradictions=contradictions)
    else
        for ta in tm.clauses
            randomize_masks!(ta; density=density, contradictions=contradictions)
        end
    end
    return tm
end

function make_binary_tm(input_len::Int; clauses::Int=20, density::Float64=0.05, states_num::Int=256)
    x = rand_input(input_len, 0.5)
    ys = Bool[true, false, true, false]
    include_limit = states_num == 256 ? 128 : states_num - 536
    include_limit = clamp(include_limit, 1, states_num - 1)
    tm = TS.TMClassifier(x, ys, clauses, 20, max(16, div(input_len, 8)), min(input_len, max(4, div(input_len, 4))), min(input_len, max(4, div(input_len, 8))); states_num=states_num, include_limit=include_limit)
    randomize_masks!(tm; density=density)
    return tm, x
end

function make_multiclass_tm(input_len::Int; classes::Int=4, clauses::Int=20, density::Float64=0.05, states_num::Int=256)
    x = rand_input(input_len, 0.5)
    ys = Int8.(collect(0:classes-1))
    include_limit = states_num == 256 ? 128 : states_num - 536
    include_limit = clamp(include_limit, 1, states_num - 1)
    tm = TS.TMClassifier(x, ys, clauses, 20, max(16, div(input_len, 8)), min(input_len, max(4, div(input_len, 4))), min(input_len, max(4, div(input_len, 8))); states_num=states_num, include_limit=include_limit)
    randomize_masks!(tm; density=density)
    return tm, x, ys
end

function first_team(tm::TS.TMClassifier)
    return typeof(first(tm.classes)) == Bool ? tm.clauses : tm.clauses[1]
end

function case!(cases::Vector{BenchCase}, group::String, name::String, make::Function, run::Function; unit::String="op", note::String="")
    push!(cases, BenchCase(group, name, make, run, unit, note))
    return cases
end

function add_input_cases!(cases::Vector{BenchCase}, opts::BenchOptions)
    sizes = opts.mode == :quick ? [784, 16_384] : opts.mode == :full ? [64, 784, 4_096, 16_384] : [64, 256, 784, 2_048, 4_096, 16_384]
    densities = opts.mode == :exhaustive ? [0.01, 0.1, 0.5, 0.9] : [0.1, 0.5]

    # Density-dependent cases: one per (size, density) pair.
    for n in sizes, d in densities
        source = rand_bitvector(n, d)
        case!(cases, "TMInput", "construct BitVector n=$n density=$d", () -> source, s -> TS.TMInput(s))
        x = TS.TMInput(source)
        case!(cases, "TMInput", "sum n=$n density=$d", () -> x, s -> sum(s))
        case!(cases, "TMInput", "getindex n=$n density=$d", () -> (x, rand(1:n, 256)), s -> begin
            y = false
            @inbounds for idx in s[2]
                y = xor(y, s[1][idx])
            end
            y
        end; unit="256 getindex")
    end

    # Density-independent cases: one per size (kept out of the density loop so
    # they are not run twice under duplicate names).
    for n in sizes
        case!(cases, "TMInput", "construct zero n=$n", () -> n, s -> TS.TMInput(s))
        case!(cases, "TMInput", "setindex n=$n", () -> (TS.TMInput(n), rand(1:n, 256)), s -> begin
            @inbounds for idx in s[2]
                s[1][idx] = true
            end
            s[1]
        end; unit="256 setindex!")
    end

    image = rand(Float32, 28, 28)
    case!(cases, "TMInput", "booleanize 28x28 one threshold", () -> image, s -> TS.booleanize(s, 0.5))
    case!(cases, "TMInput", "booleanize 28x28 four thresholds", () -> image, s -> TS.booleanize(s, 0.0, 0.25, 0.5, 0.75))
    return cases
end

function add_clause_cases!(cases::Vector{BenchCase}, opts::BenchOptions)
    if opts.mode == :quick
        sizes = [784, 16_384]
        literal_densities = [0.02, 0.5]
    elseif opts.mode == :full
        sizes = [64, 784, 4_096, 16_384]
        literal_densities = [0.01, 0.05, 0.25, 0.5]
    else
        sizes = [64, 256, 784, 2_048, 4_096, 16_384]
        literal_densities = [0.001, 0.01, 0.05, 0.25, 0.5]
    end

    for n in sizes, d in literal_densities
        case!(cases, "Clause", "check dense n=$n literals=$d", () -> begin
            tm, x = make_binary_tm(n; clauses=8, density=d)
            ta = first_team(tm)
            (tm, x, @view(ta.positive_included_literals[:, 1]), @view(ta.positive_included_literals_inverted[:, 1]))
        end, s -> TS.check_clause(s[1], s[2], s[3], s[4]))

        case!(cases, "Clause", "update index n=$n literals=$d", () -> begin
            tm, _ = make_binary_tm(n; clauses=8, density=d)
            ta = first_team(tm)
            (tm, @view(ta.positive_included_literals[:, 1]), @view(ta.positive_included_literals_inverted[:, 1]), @view(ta.positive_included_literals_idx[:, 1]))
        end, s -> TS.update_index(s[1], s[2], s[3], s[4]))

        case!(cases, "Clause", "check sparse-index n=$n literals=$d", () -> begin
            tm, x = make_binary_tm(n; clauses=8, density=d)
            ta = first_team(tm)
            l = @view(ta.positive_included_literals[:, 1])
            li = @view(ta.positive_included_literals_inverted[:, 1])
            idx = @view(ta.positive_included_literals_idx[:, 1])
            TS.update_index(tm, l, li, idx)
            (tm, x, l, li, idx)
        end, s -> TS.check_clause(s[1], s[2], s[3], s[4], s[5]))

        case!(cases, "Clause", "include_literals_sum n=$n literals=$d", () -> begin
            tm, _ = make_binary_tm(n; clauses=8, density=d)
            ta = first_team(tm)
            (@view(ta.positive_included_literals[:, 1]), @view(ta.positive_included_literals_inverted[:, 1]), length(ta.positive_included_literals[:, 1]))
        end, s -> TS.include_literals_sum(s[1], s[2], s[3]))
    end
    return cases
end

function add_model_cases!(cases::Vector{BenchCase}, opts::BenchOptions)
    configs = opts.mode == :quick ?
        [(784, 10, 20, 0.05), (16_384, 4, 8, 0.02)] :
        [(64, 4, 8, 0.05), (784, 10, 20, 0.05), (4_096, 16, 32, 0.02), (16_384, 32, 64, 0.02)]
    if opts.mode == :exhaustive
        append!(configs, [(16_384, 65, 64, 0.5), (16_384, 65, 64, 0.02)])
    end

    for (n, classes, clauses, d) in configs
        case!(cases, "Model", "vote binary n=$n clauses=$clauses literals=$d", () -> begin
            tm, x = make_binary_tm(n; clauses=clauses, density=d)
            (tm, first_team(tm), x)
        end, s -> TS.vote(s[1], s[2], s[3], index=false))

        case!(cases, "Model", "predict binary n=$n clauses=$clauses literals=$d", () -> make_binary_tm(n; clauses=clauses, density=d), s -> TS.predict(s[1], s[2]))

        case!(cases, "Model", "train binary n=$n clauses=$clauses", () -> make_binary_tm(n; clauses=clauses, density=0.0), s -> TS.train!(s[1], s[2], true))

        case!(cases, "Model", "predict multiclass n=$n classes=$classes clauses=$clauses literals=$d", () -> make_multiclass_tm(n; classes=classes, clauses=clauses, density=d), s -> TS.predict(s[1], s[2]))

        case!(cases, "Model", "train multiclass n=$n classes=$classes clauses=$clauses", () -> make_multiclass_tm(n; classes=classes, clauses=clauses, density=0.0), s -> TS.train!(s[1], s[2], first(s[3])))

        case!(cases, "Model", "predict batch n=$n classes=$classes clauses=$clauses literals=$d", () -> begin
            tm, _, _ = make_multiclass_tm(n; classes=classes, clauses=clauses, density=d)
            xs = [rand_input(n, 0.5) for _ in 1:32]
            (tm, xs)
        end, s -> TS.predict(s[1], s[2]); unit="32 predictions")
    end

    case!(cases, "Model", "accuracy 4096 labels", () -> begin
        y = rand(Bool, 4096)
        pred = copy(y)
        (pred, y)
    end, s -> TS.accuracy(s[1], s[2]))

    if opts.include_io
        case!(cases, "Model", "compile multiclass 784", () -> begin
            tm, _, _ = make_multiclass_tm(784; classes=10, clauses=20, density=0.05)
            tm
        end, s -> TS.compile(s))

        case!(cases, "Model", "save/load compiled multiclass 784", () -> begin
            tm, _, _ = make_multiclass_tm(784; classes=10, clauses=20, density=0.05)
            compiled = TS.compile(tm)
            path = tempname() * ".tm"
            (compiled, path)
        end, s -> quiet(() -> begin
            TS.save(s[1], s[2])
            TS.load(s[2])
            rm(s[2], force=true)
        end))
    end

    case!(cases, "Model", "literals_count team 16384", () -> begin
        tm, _ = make_binary_tm(16_384; clauses=16, density=0.05)
        first_team(tm)
    end, s -> TS.literals_count(s))

    case!(cases, "Model", "literals_sum multiclass 16384", () -> begin
        tm, _, _ = make_multiclass_tm(16_384; classes=8, clauses=16, density=0.05)
        tm
    end, s -> TS.literals_sum(s))

    if opts.include_builtin_benchmark
        case!(cases, "Model", "TS.benchmark smoke", () -> begin
            tm, _, ys = make_multiclass_tm(64; classes=4, clauses=8, density=0.05)
            xs = [rand_input(64, 0.5) for _ in 1:16]
            labels = [ys[rand(1:length(ys))] for _ in 1:16]
            (TS.compile(tm), xs, labels)
        end, s -> quiet(() -> TS.benchmark(s[1], s[2], s[3], 1; warmup=false)))
    end
    return cases
end

function add_explain_cases!(cases::Vector{BenchCase}, opts::BenchOptions)
    sizes = opts.mode == :quick ? [784] : [784, 16_384]
    for n in sizes
        case!(cases, "Explain", "explain literal matrix n=$n", () -> begin
            tm, _ = make_binary_tm(n; clauses=16, density=0.05)
            ta = first_team(tm)
            (ta.positive_included_literals, ta.clause_size)
        end, s -> explain(s[1], s[2]))

        case!(cases, "Explain", "explain tm n=$n", () -> begin
            tm, _ = make_binary_tm(n; clauses=16, density=0.05)
            tm
        end, s -> explain(s))

        case!(cases, "Explain", "explain tm input n=$n", () -> begin
            tm, x = make_binary_tm(n; clauses=16, density=0.05)
            (tm, x)
        end, s -> explain(s[1], s[2]))
    end
    return cases
end

function add_fastconv_cases!(cases::Vector{BenchCase}, opts::BenchOptions)
    specs = opts.mode == :quick ?
        [((64,), (5,)), ((16, 16), (3, 3))] :
        [((64,), (5,)), ((256,), (9,)), ((16, 16), (3, 3)), ((32, 32), (5, 5)), ((12, 12, 4), (3, 3, 2))]

    for (esize, ksize) in specs
        label = join(esize, "x") * " kernel " * join(ksize, "x")
        case!(cases, "FastConv", "fastconv $label", () -> begin
            E = rand(Float32, esize...)
            k = rand(Float32, ksize...)
            (E, k)
        end, s -> fastconv(s[1], s[2]))
    end
    return cases
end

function add_hdc_cases!(cases::Vector{BenchCase}, opts::BenchOptions)
    opts.include_hdc || return cases
    contexts = opts.mode == :quick ? [8, 64] : opts.mode == :full ? [8, 64, 256] : [2, 8, 32, 64, 128, 256]
    tokens = UInt8.(collect("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;:\n"))
    hv_density = round(Int, HV_DIMENSIONS * (1 - 0.5^(1 / NGRAM)))

    case!(cases, "TEXT HDC", "random_hv dim=$HV_DIMENSIONS", () -> hv_density, k -> random_hv(HV_DIMENSIONS, k))

    case!(cases, "TEXT HDC", "gen_ngram ngram=$NGRAM", () -> begin
        hvectors = Dict(t => random_hv(HV_DIMENSIONS, hv_density) for t in tokens)
        scratch = BitVector(undef, HV_DIMENSIONS)
        scratch2 = BitVector(undef, HV_DIMENSIONS)
        context = tokens[1:NGRAM]
        (hvectors, context, scratch, scratch2)
    end, s -> gen_ngram(s[1], s[2], s[3], s[4]))

    for len in contexts
        case!(cases, "TEXT HDC", "gen_context_hvector len=$len", () -> begin
            hvectors = Dict(t => random_hv(HV_DIMENSIONS, hv_density) for t in tokens)
            acc = zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS)
            scratch = BitVector(undef, HV_DIMENSIONS)
            scratch2 = BitVector(undef, HV_DIMENSIONS)
            context = [tokens[rand(1:length(tokens))] for _ in 1:len]
            (acc, scratch, scratch2, context, hvectors)
        end, s -> gen_context_hvector!(s[1], s[2], s[3], s[4], s[5]))
    end
    return cases
end

# Drop cases whose (group, name, unit) collide. Such cases are indistinguishable
# downstream (the dashboard keys results by exactly this tuple), so a duplicate is
# either redundant work or a hidden case. Warn loudly and keep the first.
function dedupe_cases(cases::Vector{BenchCase})::Vector{BenchCase}
    seen = Set{Tuple{String,String,String}}()
    unique_cases = BenchCase[]
    for c in cases
        key = (c.group, c.name, c.unit)
        if key in seen
            @warn "Dropping duplicate benchmark case (name collides with an earlier case)" group=c.group name=c.name unit=c.unit
            continue
        end
        push!(seen, key)
        push!(unique_cases, c)
    end
    return unique_cases
end

function build_cases(opts::BenchOptions)::Vector{BenchCase}
    Random.seed!(opts.seed)
    cases = BenchCase[]
    add_input_cases!(cases, opts)
    add_clause_cases!(cases, opts)
    add_model_cases!(cases, opts)
    add_explain_cases!(cases, opts)
    add_fastconv_cases!(cases, opts)
    add_hdc_cases!(cases, opts)
    return dedupe_cases(cases)
end

function fmt_ns(ns::Real)::String
    if ns < 1_000
        return @sprintf("%.1f ns", ns)
    elseif ns < 1_000_000
        return @sprintf("%.2f us", ns / 1_000)
    elseif ns < 1_000_000_000
        return @sprintf("%.2f ms", ns / 1_000_000)
    else
        return @sprintf("%.2f s", ns / 1_000_000_000)
    end
end

function markdown_table(results::Vector{BenchResult}, opts::BenchOptions)::String
    io = IOBuffer()
    println(io, "# Tsetlin.jl Benchmark Results")
    println(io)
    println(io, "- Date: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
    println(io, "- Julia threads: $(Threads.nthreads())")
    println(io, "- Mode: `$(opts.mode)`")
    println(io, "- Repeats: $(opts.repeats)")
    println(io, "- Minimum sample time: $(round(Int, opts.min_sample_ns / 1_000_000)) ms")
    println(io)
    println(io, "| Group | Benchmark | Median/op | Min/op | Max/op | Loops | Allocated | Unit | Note |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in results
        println(io, "| $(r.group) | $(r.name) | $(fmt_ns(r.median_ns)) | $(fmt_ns(r.min_ns)) | $(fmt_ns(r.max_ns)) | $(r.loops) | $(r.allocated_bytes) | $(r.unit) | $(r.note) |")
    end
    return String(take!(io))
end

csv_escape(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""

function csv_table(results::Vector{BenchResult})::String
    io = IOBuffer()
    println(io, "group,name,unit,loops,repeats,median_ns,min_ns,max_ns,allocated_bytes,note")
    for r in results
        println(io, join((
            csv_escape(r.group),
            csv_escape(r.name),
            csv_escape(r.unit),
            r.loops,
            r.repeats,
            @sprintf("%.3f", r.median_ns),
            @sprintf("%.3f", r.min_ns),
            @sprintf("%.3f", r.max_ns),
            r.allocated_bytes,
            csv_escape(r.note),
        ), ","))
    end
    return String(take!(io))
end

json_escape(x) = replace(string(x), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
json_string(x) = "\"" * json_escape(x) * "\""

function json_results(results::Vector{BenchResult}, opts::BenchOptions)::String
    io = IOBuffer()
    println(io, "{")
    println(io, "  \"mode\": $(json_string(opts.mode)),")
    println(io, "  \"repeats\": $(opts.repeats),")
    println(io, "  \"min_sample_ns\": $(opts.min_sample_ns),")
    println(io, "  \"threads\": $(Threads.nthreads()),")
    println(io, "  \"results\": [")
    for (i, r) in enumerate(results)
        comma = i == length(results) ? "" : ","
        median_ns = @sprintf("%.3f", r.median_ns)
        min_ns = @sprintf("%.3f", r.min_ns)
        max_ns = @sprintf("%.3f", r.max_ns)
        println(io, "    {\"group\": $(json_string(r.group)), \"name\": $(json_string(r.name)), \"unit\": $(json_string(r.unit)), \"loops\": $(r.loops), \"repeats\": $(r.repeats), \"median_ns\": $median_ns, \"min_ns\": $min_ns, \"max_ns\": $max_ns, \"allocated_bytes\": $(r.allocated_bytes), \"note\": $(json_string(r.note))}$comma")
    end
    println(io, "  ]")
    println(io, "}")
    return String(take!(io))
end

function render(results::Vector{BenchResult}, opts::BenchOptions)::String
    opts.format == :markdown && return markdown_table(results, opts)
    opts.format == :csv && return csv_table(results)
    opts.format == :json && return json_results(results, opts)
    error("Unsupported format: $(opts.format)")
end

function main(args=ARGS)
    opts = parse_args(args)
    Random.seed!(opts.seed)
    cases = build_cases(opts)
    println("Prepared $(length(cases)) benchmark cases in $(opts.mode) mode.")
    println("Running with $(Threads.nthreads()) Julia thread(s).")
    results = run_cases(cases, opts)
    output = render(results, opts)
    if isempty(opts.output)
        println()
        print(output)
    else
        outdir = dirname(opts.output)
        isempty(outdir) || mkpath(outdir)
        write(opts.output, output)
        println()
        println("Wrote benchmark results to $(opts.output)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
