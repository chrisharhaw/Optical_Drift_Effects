# scripts/plot_generic_cusp.jl
using Optical_Drift_Effects
using StaticArrays
using LinearAlgebra
using Plots
using Contour  

# -----------------------------
# USER: construct your lens here
# -----------------------------
# Example:
# lens = GenericCusp( ... )
# or whatever your constructor is

lens = generic_cusp(1, 1)   # <-- change this line to your lens object / parameters

# -----------------------------
# Grid settings
# -----------------------------
Nx, Ny = 300, 300
xmin, xmax = -4.0, 0.5
ymin, ymax = -2.0, 2.0

xs = range(xmin, xmax; length=Nx)
ys = range(ymin, ymax; length=Ny)

# Storage
Ψ   = zeros(Float64, Ny, Nx)   # potential
ax  = zeros(Float64, Ny, Nx)   # deflection x
ay  = zeros(Float64, Ny, Nx)   # deflection y
detJ = zeros(Float64, Ny, Nx)  # lensing jacobian

# -----------------------------
# USER: pointwise evaluation functions
# Replace these wrappers with your actual function names
# -----------------------------
potential_at(lens, θ) = potential(lens, θ)                 
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)
jacobian_at(lens, θ) = deflection_jacobian(lens, θ)        # returns 2x2 (SMatrix ok)

# -----------------------------
# Evaluate on grid
# -----------------------------
for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
    θ = @SVector [x, y]

    Ψ[i, j] = potential_at(lens, θ)

    a = deflection_at(lens, θ)
    ax[i, j] = a[1]
    ay[i, j] = a[2]

    J = jacobian_at(lens, θ)
    detJ[i, j] = det(Matrix(J))   # safe if J is SMatrix or Matrix

    betax = x - ax[i, j]
    betay = y - ay[i, j]
    beta = @SVector [betax, betay]
end

# -----------------------------
# Plot: Potential
# -----------------------------
p1 = heatmap(xs, ys, Ψ';
    aspect_ratio=:equal,
    title="Lensing potential Ψ(θ)",
    xlabel="θx", ylabel="θy",
    colorbar_title="Ψ"
)

# -----------------------------
# Plot: Deflection components
# -----------------------------
p2 = heatmap(ys, xs, ax;
    aspect_ratio=:equal,
    title="Deflection ax(θ)",
    xlabel="θx", ylabel="θy",
    colorbar_title="ax"
)

p3 = heatmap(ys, xs, ay;
    aspect_ratio=:equal,
    title="Deflection ay(θ)",
    xlabel="θx", ylabel="θy",
    colorbar_title="ay"
)

# -----------------------------
# Plot: det(J)
# -----------------------------
p4 = heatmap(ys, xs, detJ';
    aspect_ratio=:equal,
    title="det(Jacobian)",
    xlabel="θx", ylabel="θy",
    colorbar_title="detJ"
)

contour!(p4, ys, xs, detJ';
    levels=[0.0],  
    linewidth=2,
    linecolor=:white,
    label="Critical curve"
)

# -----------------------------
# Plot: caustic curves (recalculate critical curves (det(J)=0) 
# -----------------------------
# You can add contour lines to p4 to show where det(J)=0, which are the critical curves. For example:

levs = [0.0]
cs = contours(xs, ys, detJ, levs)

critical_polylines = Vector{Vector{SVector{2,Float64}}}()


for lvl in levels(cs)
    for line in lines(lvl)
        xline, yline = coordinates(line)   # <- two vectors
        poly = [@SVector [xline[k], yline[k]] for k in eachindex(xline)]
        push!(critical_polylines, poly)
    end
end

caustic_polylines = Vector{Vector{SVector{2,Float64}}}()

for poly in critical_polylines
    mapped = [θ - deflection_at(lens, θ) for θ in poly]
    push!(caustic_polylines, mapped)
end

p5 = plot(; aspect_ratio=:equal,
    title="Caustic curves in source plane",
    xlabel="βx", ylabel="βy",
    legend=false
)

for poly in caustic_polylines
    plot!(p5, first.(poly), last.(poly), lw=2)
end

# -----------------------------
# Display / save
# -----------------------------
plot(p1, p4, p5; layout=(1,3), size=(1400, 800))
savefig("generic_cusp_fields.png")
println("Saved: generic_cusp_fields.png")



# -----------------------------
# Image configuration 
# -----------------------------

# --- helper: real cube root ---
cbrt_real(x::Real) = sign(x) * abs(x)^(1/3)

# --- discriminant field Δ(β) for the generic cusp ---
function cusp_discriminant(lens, βx::Float64, βy::Float64)
    d = lens.d
    e = lens.e
    a = e - 0.5*d^2               # coefficient of y^3
    b = d * βx                    # coefficient of y
    c = -βy                       # constant term

    p = b / a
    q = c / a

    Δ = (q/2)^2 + (p/3)^3
    return Δ
end

# --- choose a source-plane window around the cusp ---
βxs = range(-6.5, 1.0; length=500)
βys = range(-8.5, 8.5; length=500)

Δgrid = Array{Float64}(undef, length(βys), length(βxs))
for (j, βy) in enumerate(βys)
    for (i, βx) in enumerate(βxs)
        Δgrid[j,i] = cusp_discriminant(lens, βx, βy)
    end
end

# --- Plot: 3-image region (Δ<0) vs 1-image region (Δ>0) ---
p_mult = contourf(
    βxs, βys, Δgrid;
    levels=[-Inf, 0.0, Inf],
    aspect_ratio=:equal,
    xlabel="βx", ylabel="βy",
    title="Image multiplicity near cusp (Δ<0 → 3 images, Δ>0 → 1 image)",
    colorbar=false,
    legend=false
)

# overlay caustic
for poly in caustic_polylines
    plot!(p_mult, first.(poly), last.(poly), lw=3)
end
savefig(p_mult, "generic_cusp_multiplicity.png")
println("Saved: generic_cusp_multiplicity.png")

# --------------------------------------------
# Image positions for a specific source position β
# ---------------------------------------------


# Solve y^3 + p y + q = 0 for real roots
function depressed_cubic_real_roots(p::Float64, q::Float64)
    Δ = (q/2)^2 + (p/3)^3

    if Δ > 0
        # one real root
        u = cbrt_real(-q/2 + sqrt(Δ))
        v = cbrt_real(-q/2 - sqrt(Δ))
        return [u + v]
    elseif abs(Δ) ≤ 1e-14
        # multiple root case (on/near caustic)
        u = cbrt_real(-q/2)
        return [2u, -u]  # (double root at -u)
    else
        # three real roots
        r = 2 * sqrt(-p/3)
        φ = acos( (3q/(2p)) * sqrt(-3/p) )
        return [
            r * cos(φ/3),
            r * cos((φ + 2π)/3),
            r * cos((φ + 4π)/3)
        ]
    end
end

function image_positions(lens, β::SVector{2,Float64})
    d = lens.d
    e = lens.e
    βx, βy = β[1], β[2]

    a = e - 0.5*d^2
    b = d * βx
    c = -βy

    p = b / a
    q = c / a

    ys = depressed_cubic_real_roots(p, q)

    # recover x from βx = x + (d/2) y^2
    imgs = SVector{2,Float64}[]
    for y in ys
        x = βx - 0.5*d*y^2
        push!(imgs, @SVector [x, y])
    end
    return imgs
end


# ---------------------------------------------
# Plot overlay figure
# ---------------------------------------------

# Choose some source positions to test image configurations
sources = SVector{2,Float64}[
    SVector{2,Float64}(-3.0, 0.0),   # often inside → 3 images
    SVector{2,Float64}(-0.5, 3.0),   # likely outside → 1 image
    SVector{2,Float64}(-2.5, 1.0),   # near boundary
]

# Build a palette with one color per source
pal = palette(:tab10, length(sources))

# Compute images for each source
all_images = Vector{Vector{SVector{2,Float64}}}()
for β in sources
    push!(all_images, image_positions(lens, β))
end

println("Image counts per source: ", [length(imgs) for imgs in all_images])

# Lens-plane panel: critical curve + images
p_lens = plot(; aspect_ratio=:equal,
    title="Lens plane: critical curve + images",
    xlabel="θx", ylabel="θy",
    legend=false
)

# plot critical curve(s)
for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly), lw=2)
end

for (k, imgs) in enumerate(all_images)
    isempty(imgs) && continue
    scatter!(p_lens, first.(imgs), last.(imgs);
        markersize=5,
        markershape=:circle,
        seriescolor=pal[k],
        markerstrokecolor=pal[k],
        markerstrokewidth=0
    )
end

# Source-plane panel: caustic + sources
p_src = plot(; aspect_ratio=:equal,
    title="Source plane: caustic + sources",
    xlabel="βx", ylabel="βy",
    legend=false
)

# plot caustic(s)
for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly), lw=2)
end

# Plot sources as crosses, using the SAME colors
for (k, β) in enumerate(sources)
    scatter!(p_src, [β[1]], [β[2]];
        markersize=9,
        markershape=:x,
        seriescolor=pal[k],
        markerstrokecolor=pal[k],
        markerstrokewidth=2
    )
end

# plot sources (crosses)
scatter!(p_src, first.(sources), last.(sources);
    markersize=7, markershape=:x
)

# Combine and save
p_overlay = plot(p_lens, p_src; layout=(1,2), size=(1200, 600))
display(p_overlay)

savefig(p_overlay, "generic_cusp_overlay.png")
println("Saved: generic_cusp_overlay.png")