# Domain-agnostic occlusion-saliency demo (no MNIST / no plotting deps).
#
# saliency(tm, x)              -> per-bit importance for ONE prediction
# class_saliency(tm, X, c)     -> averaged per-bit importance for class c
#
# Both return a plain Vector{Float64} over input bits, so they work for any
# input modality (images, text/HDC vectors, tabular bits). Reshape only when
# you want to render (e.g. reshape(map, 28, 28) for MNIST).

include(joinpath(@__DIR__, "..", "src", "utils", "explain.jl"))
using .Tsetlin
using Random

# --- a tiny learnable task: class c is signalled by a fixed set of bits ---
const N, CLASSES, SIG = 256, 4, 6
rng = MersenneTwister(1)
perm = randperm(rng, N)
signatures = [perm[(c - 1) * SIG + 1 : c * SIG] for c in 1:CLASSES]

function gen(count)
    X = TMInput[]; Y = Int8[]
    for _ in 1:count
        c = rand(rng, 0:CLASSES - 1)
        bv = falses(N)
        @inbounds for i in 1:N; bv[i] = rand(rng) < 0.1; end      # background noise
        for i in signatures[c + 1]; bv[i] = rand(rng) < 0.95; end # class signal
        push!(X, TMInput(bv)); push!(Y, Int8(c))
    end
    return X, Y
end

Xtr, Ytr = gen(4000); Xte, Yte = gen(800)
ys = Int8.(collect(0:CLASSES - 1))
tm = TMClassifier(Xtr[1], ys, 40, 15, 20, 12, 1; states_num=256, include_limit=128)
train!(tm, Xtr, Ytr, Xte, Yte, 30; exclusive=true, verbose=0)
println("accuracy: ", round(accuracy(predict(tm, Xte), Yte) * 100, digits=1), "%\n")

# --- per-instance: which bits drove THIS prediction ---
x = Xte[1]
imp = saliency(tm, x)                       # Vector{Float64}, length == length(x)
topk = sortperm(imp, rev=true)[1:SIG]
println("instance: predicted=$(predict(tm, x)) true=$(Yte[1])")
println("  top saliency bits : ", sort(topk))
println("  class signature    : ", sort(signatures[Int(Yte[1]) + 1]))

# --- per-class: averaged attribution map ---
println("\nper-class saliency (top-$SIG bits should match the signature):")
for c in Int8.(0:CLASSES - 1)
    Xc = [Xte[i] for i in eachindex(Xte) if Yte[i] == c]
    smap, used = class_saliency(tm, Xc, c; limit=100)
    top = sort(sortperm(smap, rev=true)[1:SIG])
    println("  class $c (n=$used): saliency ", top, "  signature ", sort(signatures[c + 1]))
end
