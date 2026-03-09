# scripts/checkerboard_lensing.jl
using Optical_Drift_Effects
using StaticArrays
using LinearAlgebra
using Plots
using Contour  
using Random

# -----------------------------
# Lens set-up
# -----------------------------
# d=1, e=1 
lens = generic_cusp(1, 1)   # <-- change this line to your lens object / parameters 

# ------------------------------
# Source set-up
# ------------------------------
cell_size = 0.2
src = CheckerboardSource(cell_size=cell_size, β0=(0.0, 0.0), window_size=2.0)

# -----------------------------
# Grid settings
# -----------------------------

os = 5           # oversampling factor (pixels per output pixel, per axis)

Nx_pix, Ny_pix = 200, 200                 # desired output (pixelated) resolution
Nx_hi,  Ny_hi  = os * Nx_pix, os * Ny_pix  # high-res ray grid

xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0

# High-res ray grid 
xs_hi = range(xmin, xmax; length=Nx_hi)
ys_hi = range(ymin, ymax; length=Ny_hi)

# Output pixel centres (for plot axes)
xs_pix = range(xmin, xmax; length=Nx_pix)
ys_pix = range(ymin, ymax; length=Ny_pix)


detJ = zeros(Float64, Ny_hi, Nx_hi)  # lensing jacobian

# ---------------------------------------------
# lensing functions at a point (wrapper for your lens functions)
# ---------------------------------------------             
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)
jacobian_at(lens, θ) = deflection_jacobian(lens, θ)        # returns 2x2 (SMatrix ok)

# ---------------------------------------------
# Evaluate on grid
# ---------------------------------------------
for (j, y) in enumerate(ys_hi), (i, x) in enumerate(xs_hi)
    θ = @SVector [x, y]

    J = jacobian_at(lens, θ)
    detJ[i, j] = det(Matrix(J))   
end

# -----------------------------
# Plot: caustic curves (recalculate critical curves (det(J)=0) 
# -----------------------------
# You can add contour lines to p4 to show where det(J)=0, which are the critical curves. For example:

levs = [0.0]
cs = contours(xs_hi, ys_hi, detJ, levs)

critical_polylines = Vector{Vector{SVector{2,Float64}}}()
caustic_polylines = Vector{Vector{SVector{2,Float64}}}()

for lvl in levels(cs)
    for line in lines(lvl)
        xline, yline = coordinates(line)   # <- two vectors
        poly = [@SVector [xline[k], yline[k]] for k in eachindex(xline)]
        push!(critical_polylines, poly)
    end
end

for poly in critical_polylines
    mapped = [θ - deflection_at(lens, θ) for θ in poly]
    push!(caustic_polylines, mapped)
end


# --------------------------------------------
# Image positions for a specific source position β
# ---------------------------------------------
# --- helper: real cube root ---
cbrt_real(x::Real) = sign(x) * abs(x)^(1/3)

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
    d, e = lens.d, lens.e
    βx, βy = β[1], β[2]

    a = e - 0.5*d^2
    b = d * βx
    c = -βy

    p = b / a
    q = c / a

    ys_roots = depressed_cubic_real_roots(p, q)

    # recover x from βx = x + (d/2) y^2
    imgs = SVector{2,Float64}[]
    for y in ys_roots
        x = βx - 0.5*d*y^2
        push!(imgs, @SVector [x, y])
    end
    return imgs
end

# ---------------------------------------------
# Ray-shooting approach for extended sources
# ---------------------------------------------

# lens equation mapping θ -> β
β_from_θ(lens, θ::SVector{2,Float64}) = θ - deflection_at(lens, θ)

"""
Return image-plane intensity map I(θ) on a grid (xcent × ycent).
I_img(θ) = I_src(β(θ)).
"""
function ray_shoot_intensity_map(lens, src, xs, ys)
    Nx = length(xs)
    Ny = length(ys)
    I = Matrix{Float64}(undef, Ny, Nx)   # (y,x) for heatmap

    @inbounds for j in 1:Ny
        y = ys[j]
        for i in 1:Nx
            x = xs[i]
            θ = SVector{2,Float64}(x, y)
            β = β_from_θ(lens, θ)
            I[j, i] = float(intensity(src, β))
        end
    end
    return I
end


#---generate ray-shooting intensity map and plot with caustics/critical curves ---

I_hi  = ray_shoot_intensity_map(lens, src, xs_hi, ys_hi)
p_lens = heatmap(xs_hi, ys_hi, log10.(I_hi .+ 1e-12);  # log for display
    aspect_ratio=:equal,
    xlabel="θx", ylabel="θy",
    title="Lens plane",
    colorbar=false, legend=false
)

for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly); lw=2, linecolor=:blue)
end

xs_src = range(xmin, xmax; length=Nx_pix)
ys_src = range(ymin, ymax; length=Ny_pix)

I_src_map = [intensity(src, SVector{2,Float64}(x, y))
             for y in ys_src, x in xs_src]   # (Ny × Nx), matching heatmap layout

p_src = heatmap(xs_src, ys_src, I_src_map;
    aspect_ratio=:equal,
    xlabel="βx", ylabel="βy",
    title="Source plane",
    colorbar=false, legend=false
)

for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly); lw=2, linecolor=:blue)
end

p_overlay = plot(p_lens, p_src; layout=(1,2), size=(1200, 600))

savefig(p_overlay, "checkerboard.png")
println("Saved: checkerboard.png")