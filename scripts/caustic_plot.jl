using StaticArrays
using LinearAlgebra
using Plots
using Contour   # contour extraction on gridded data

# -------------------------
# Cusp map: (u,v) -> (β1,β2)
# -------------------------
@inline function cusp_map(d::Int, e::Int, uv::SVector{2,Float64})
    u, v = uv
    β1 = u + d * v^2 / 2
    β2 = d * v * u + e * v^3
    return @SVector [β1, β2]
end

# Jacobian of cusp_map wrt (u,v)
@inline function cusp_jacobian(d::Int, e::Int, uv::SVector{2,Float64})
    u, v = uv
    # β1 = u + (d/2) v^2
    # β2 = d v u + e v^3
    # ∂β/∂(u,v) =
    # [ 1        d v ]
    # [ d v   d u + 3 e v^2 ]
    return @SMatrix [ 1.0          d*v
                      d*v   d*u + 3*e*v^2 ]
end

# -------------------------
# Grid in (u,v)
# -------------------------
Nu, Nv = 400, 400
umin, umax = -2.0, 2.0
vmin, vmax = -2.0, 2.0

us = range(umin, umax; length=Nu)
vs = range(vmin, vmax; length=Nv)

detJ = zeros(Float64, Nv, Nu)  # note: rows=v, cols=u

for (j, v) in enumerate(vs), (i, u) in enumerate(us)
    uv = SVector{2,Float64}(u, v)
    J = cusp_jacobian(1, 1, uv)        # <-- set d,e here
    detJ[j,i] = det(J)
end

d, e = 1, 1  # choose ±1, ±1

# -------------------------
# Extract critical curve detJ=0 in (u,v)
# -------------------------
# Contour.jl expects x-grid, y-grid, z-matrix with z[y_index, x_index]
# levs = [0.0]
# cs = contours(us, vs, detJ, levs)

#  Collect polylines in (u,v)
# critical_polylines = Vector{Vector{SVector{2,Float64}}}()
# for c in levels(cs)                 # each contour level (only 0.0)
#     for line in lines(c)            # each connected contour line
#         pts = coordinates(line)     # tuples (x=u, y=v)
#         poly = [SVector{2,Float64}(p[1], p[2]) for p in pts]
#         push!(critical_polylines, poly)
#     end
# end

levs = [0.0]
cs = contours(us, vs, detJ, levs)

critical_polylines = Vector{Vector{SVector{2,Float64}}}()

for lvl in Contour.levels(cs)          # contour objects at each level
    for line in Contour.lines(lvl)     # each connected polyline
        pts = Contour.coordinates(line)  # Vector of (u,v) tuples
        poly = [SVector{2,Float64}(p[1], p[2]) for p in pts]
        push!(critical_polylines, poly)
    end
end

# -------------------------
# Map critical curve -> caustic in (β1,β2)
# -------------------------
caustic_polylines = Vector{Vector{SVector{2,Float64}}}()
for poly in critical_polylines
    mapped = [cusp_map(d, e, uv) for uv in poly]
    push!(caustic_polylines, mapped)
end

# -------------------------
# Plot (u,v) detJ field + critical curve
# -------------------------
p_uv = heatmap(us, vs, detJ;
    aspect_ratio=:equal,
    title="det(J) in (u,v) with critical curve det(J)=0",
    xlabel="u", ylabel="v",
    colorbar_title="detJ"
)
for poly in critical_polylines
    plot!(p_uv, [p[1] for p in poly], [p[2] for p in poly], lw=2, color=:white, label=false)
end

# -------------------------
# Plot caustic in (β1,β2)
# -------------------------
p_beta = plot(; aspect_ratio=:equal,
    title="Caustic curve in source plane (β1, β2)",
    xlabel="β1", ylabel="β2",
    legend=false
)
for poly in caustic_polylines
    plot!(p_beta, [p[1] for p in poly], [p[2] for p in poly], lw=2)
end

plot(p_uv, p_beta; layout=(1,2), size=(1200, 500))
savefig("cusp_critical_caustic.png")
println("Saved: cusp_critical_caustic.png")