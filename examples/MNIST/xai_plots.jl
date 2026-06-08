# Shared rendering helpers for the MNIST XAI examples (28x28).
# Overlays put attributions on top of a faint version of the input digit so the
# explanation has spatial context; near-zero values are transparent.
using CairoMakie

img28(v) = reverse(reshape(Float64.(v), 28, 28), dims=2)
const SIGNED_CMAP = cgrad([:dodgerblue, :black, :red])      # blue=IS-NOT, black=neutral, red=IS
const CLAUSE_CMAP = cgrad([:orangered, :black, :springgreen])  # red=violated, green=satisfied

nodec!(ax) = (hidedecorations!(ax); hidespines!(ax))
faint!(ax, xb; scale=0.28) = heatmap!(ax, img28(xb) .* scale, colormap=:grays, colorrange=(0, 1))
gray!(ax, xb) = (heatmap!(ax, img28(xb), colormap=:grays); nodec!(ax))

# signed attribution (red=supports, blue=against, dark=neutral); thresh hides near-zero
function signed!(ax, v; thresh=0.12, underlay=nothing)
    underlay === nothing || faint!(ax, underlay)
    m = maximum(abs, v); m = m == 0 ? 1.0 : m
    a = map(t -> abs(t) < thresh * m ? NaN : t, v)
    heatmap!(ax, img28(a), colormap=SIGNED_CMAP, colorrange=(-m, m), nan_color=:transparent)
    nodec!(ax)
end

# positive-only "what supports the class"
function support!(ax, v; underlay=nothing)
    underlay === nothing || faint!(ax, underlay)
    a = map(t -> t <= 0 ? NaN : t, v)
    heatmap!(ax, img28(a), colormap=:hot, nan_color=:transparent); nodec!(ax)
end

# counterfactual flips over the dimmed digit: removed (was on) = blue, added = red
function cf_overlay!(ax, xb, flips)
    faint!(ax, xb; scale=0.35)
    v = fill(NaN, length(xb))
    for f in flips; v[f] = xb[f] > 0.5 ? -1.0 : 1.0; end
    heatmap!(ax, img28(v), colormap=SIGNED_CMAP, colorrange=(-1, 1), nan_color=:transparent); nodec!(ax)
end

# clause as a logical rule over the faint digit: satisfied literals = green, violated = red
function clause_overlay!(ax, xb, matched, failed)
    faint!(ax, xb)
    v = fill(NaN, length(xb))
    @inbounds for i in eachindex(v)
        failed[i] && (v[i] = -1.0)
        matched[i] && (v[i] = 1.0)
    end
    heatmap!(ax, img28(v), colormap=CLAUSE_CMAP, colorrange=(-1, 1), nan_color=:transparent); nodec!(ax)
end
