# scripts/plot_generic_cusp.jl
using Optical_Drift_Effects
using StaticArrays
using LinearAlgebra
using Plots
using Contour  
using Random

# -----------------------------
# Lens set-up
# -----------------------------

lens = generic_cusp(1, 1)   # <-- change this line to your lens object / parameters

# ------------------------------
# Source set-up
# ------------------------------

# Source circle parameters
β0 = SVector{2,Float64}(-3.0, 0.2)   # center in source plane 
Rs = 0.15                             # source radius 
N = 10000                              # interior sampling number of points (if used)

Nϕ = 200                           # boundary sampling points for image curve tracking (if used)
ϕs = range(0, 2π; length=Nϕ+1)[1:end-1]
source_boundary = [SVector{2,Float64}(β0[1] + Rs*cos(ϕ), β0[2] + Rs*sin(ϕ)) for ϕ in ϕs]

# -----------------------------
# Grid settings
# -----------------------------
Nx, Ny = 300, 300
xmin, xmax = -8.0, 0.5
ymin, ymax = -4.0, 4.0

xs = range(xmin, xmax; length=Nx)
ys = range(ymin, ymax; length=Ny)

# Storage
Ψ   = zeros(Float64, Ny, Nx)   # potential
ax  = zeros(Float64, Ny, Nx)   # deflection x
ay  = zeros(Float64, Ny, Nx)   # deflection y
detJ = zeros(Float64, Ny, Nx)  # lensing jacobian

# -----------------------------
# lensing functions at a point (wrapper for your lens functions)
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
savefig("generic_cusp_fields.pdf")
println("Saved: generic_cusp_fields.pdf")


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
# Overlay figure - Extended sources
# USES BOUNDARY TRACKING TO PLOT IMAGE CURVES - first approach, not robust to caustic crossings
#
# Note: the image tracking here is very naive and will break if the source crosses a caustic.
# For a more robust implementation, you would need to handle changes in image count and track branches more carefully.
# ---------------------------------------------

function track_image_branches(lens, boundary::Vector{SVector{2,Float64}}; max_branches=3)
    # branches[b] is a Vector{SVector{2}} curve in lens plane
    branches = [SVector{2,Float64}[] for _ in 1:max_branches]

    # initialize
    imgs0 = image_positions(lens, boundary[1])
    imgs0 = sort(imgs0; by = p -> p[2])  # sort by y for stable ordering
    for b in 1:min(length(imgs0), max_branches)
        push!(branches[b], imgs0[b])
    end
    active = min(length(imgs0), max_branches)

    prev = imgs0

    # walk around boundary
    for n in 2:length(boundary)
        imgs = image_positions(lens, boundary[n])

        # if image count changes, the simple tracker will struggle.
        # For now, we’ll just handle the common case: count stays same.
        if length(imgs) != length(prev)
            # break continuity: push NaN separators into all branches
            for b in 1:max_branches
                push!(branches[b], SVector{2,Float64}(NaN, NaN))
            end
            prev = imgs
            continue
        end

        # greedy nearest-neighbor matching between prev and imgs
        used = falses(length(imgs))
        new_order = Vector{SVector{2,Float64}}(undef, length(imgs))

        for i in 1:length(prev)
            # find closest unused img to prev[i]
            best_j = 0
            best_d = Inf
            for j in 1:length(imgs)
                used[j] && continue
                d = sum((imgs[j] .- prev[i]).^2)
                if d < best_d
                    best_d = d
                    best_j = j
                end
            end
            used[best_j] = true
            new_order[i] = imgs[best_j]
        end

        # append to branches
        active = min(length(new_order), max_branches)
        for b in 1:active
            push!(branches[b], new_order[b])
        end
        prev = new_order
    end

    return branches
end

image_branches = track_image_branches(lens, source_boundary)

pal_img = palette(:tab10, 3)

# Lens plane panel
p_lens_ext = plot(; aspect_ratio=:equal,
    title="Lens plane: critical curve + extended images",
    xlabel="θx", ylabel="θy",
    legend=false
)

for poly in critical_polylines
    plot!(p_lens_ext, first.(poly), last.(poly), lw=2)
end

for (b, curve) in enumerate(image_branches)
    isempty(curve) && continue
    plot!(p_lens_ext, first.(curve), last.(curve); lw=3, seriescolor=pal_img[b])
end

# Source plane panel
p_src_ext = plot(; aspect_ratio=:equal,
    title="Source plane: caustic + source circle",
    xlabel="βx", ylabel="βy",
    legend=false
)

for poly in caustic_polylines
    plot!(p_src_ext, first.(poly), last.(poly), lw=2)
end

plot!(p_src_ext, first.(source_boundary), last.(source_boundary); lw=3, linecolor=:black)
# scatter!(p_src_ext, [β0[1]], [β0[2]]; markershape=:x, markersize=8)

p_overlay_ext = plot(p_lens_ext, p_src_ext; layout=(1,2), size=(1200, 600))
display(p_overlay_ext)

savefig(p_overlay_ext, "generic_cusp_extended_overlay.pdf")
println("Saved: generic_cusp_extended_overlay.pdf")


# ---------------------------------------------
# Overlay figure - Extended sources
# SOURCE PLANE SAMPLING APPROACH (more robust to caustic crossings, but no image curve plotting)
# Note: this approach just plots discrete image points for a grid of source positions, so it won't give smooth curves. 
# You would need a more sophisticated sampling strategy to get nice curves.
# ---------------------------------------------

"""
    sample_disk(β0, R; N=5000, rng=Random.default_rng())

Uniformly sample N points inside a disk of radius R centered at β0.
Returns Vector{SVector{2,Float64}}.
"""
function sample_disk_cloud(β0::SVector{2,Float64}, R::Float64; N::Int=5000, rng=Random.default_rng())
    pts = Vector{SVector{2,Float64}}(undef, N)
    for n in 1:N
        # Uniform in area: radius = R*sqrt(u)
        u = rand(rng)
        r = R * sqrt(u)
        ϕ = 2π * rand(rng)
        pts[n] = SVector{2,Float64}(β0[1] + r*cos(ϕ), β0[2] + r*sin(ϕ))
    end
    return pts
end

"""
    lensed_point_cloud(lens, βpts, image_positions)

For each β in βpts, compute image positions θ via `image_positions(lens, β)`.
Returns:
- θpts::Vector{SVector{2,Float64}} (all image points concatenated)
"""
function lensed_point_cloud(lens, βpts::Vector{SVector{2,Float64}}, image_positions_fn)
    θpts = SVector{2,Float64}[]
    for β in βpts
        imgs = image_positions_fn(lens, β)
        append!(θpts, imgs)
    end
    return θpts
end


βpts = sample_disk_cloud(β0, Rs; N=N)
θpts = lensed_point_cloud(lens, βpts, image_positions)

# ---- Lens plane plot ----
p_lens = plot(; aspect_ratio=:equal,
    title="Lens plane: critical curve + lensed image cloud",
    xlabel="θx", ylabel="θy",
    legend=false
)

for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly), lw=2)
end

# scatter lensed image points
scatter!(p_lens, first.(θpts), last.(θpts);
    markersize=1.5, markerstrokewidth=0, alpha=0.35
)

# ---- Source plane plot ----
p_src = plot(; aspect_ratio=:equal,
    title="Source plane: caustic + sampled source disk",
    xlabel="βx", ylabel="βy",
    legend=false
)

for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly), lw=2)
end

scatter!(p_src, first.(βpts), last.(βpts);
    markersize=1.5, markerstrokewidth=0, alpha=0.35
)
# scatter!(p_src, [β0[1]], [β0[2]];
#     markershape=:x, markersize=9
# )

p_overlay = plot(p_lens, p_src; layout=(1,2), size=(1200, 600))
display(p_overlay)

savefig(p_overlay, "generic_cusp_extended_cloud.pdf")
println("Saved: generic_cusp_extended_cloud.pdf")


# ---------------------------------------------
# Binning of points for surface density estimation 
# ---------------------------------------------

function sample_disk_hist(β0::SVector{2,Float64}, R::Float64; N::Int=10_000)
    pts = Vector{SVector{2,Float64}}(undef, N)
    for n in 1:N
        # uniform in area
        r = R * sqrt(rand())
        ϕ = 2π * rand()
        pts[n] = SVector{2,Float64}(β0[1] + r*cos(ϕ), β0[2] + r*sin(ϕ))
    end
    return pts
end


function lensed_points(lens, βpts::Vector{SVector{2,Float64}})
    θpts = SVector{2,Float64}[]
    for β in βpts
        imgs = image_positions(lens, β)   # <-- uses your point-source solver
        append!(θpts, imgs)
    end
    return θpts
end

function hist2d(points::Vector{SVector{2,Float64}},
                xedges::AbstractVector{<:Real},
                yedges::AbstractVector{<:Real})
    nx = length(xedges) - 1
    ny = length(yedges) - 1
    H = zeros(Float64, ny, nx)  # IMPORTANT: (y,x) for Plots heatmap

    xmin, xmax = xedges[1], xedges[end]
    ymin, ymax = yedges[1], yedges[end]

    for p in points
        x, y = p[1], p[2]
        # skip points outside plotting window
        if x < xmin || x >= xmax || y < ymin || y >= ymax
            continue
        end
        ix = searchsortedlast(xedges, x)
        iy = searchsortedlast(yedges, y)
        # clamp to valid bins
        ix = clamp(ix, 1, nx)
        iy = clamp(iy, 1, ny)
        H[iy, ix] += 1.0
    end
    return H
end

βpts = sample_disk_hist(β0, Rs; N)
θpts = lensed_points(lens, βpts)

Nbinsx, Nbinsy = 450, 450
xedges = range(xmin, xmax; length=Nbinsx+1)
yedges = range(ymin, ymax; length=Nbinsy+1)

H = hist2d(θpts, xedges, yedges)

# bin centers for plotting axes
xcent = @. 0.5*(xedges[1:end-1] + xedges[2:end])
ycent = @. 0.5*(yedges[1:end-1] + yedges[2:end])

# Optional: log stretch for visibility
Hplot = log10.(H .+ 1.0)

p_sb = heatmap(xcent, ycent, Hplot;
    aspect_ratio=:equal,
    xlabel="θx", ylabel="θy",
    title="Extended source lensed image (binned point cloud)",
    colorbar_title="log10(count+1)"
)

# Overlay critical curve (lens plane)
for poly in critical_polylines
    plot!(p_sb, first.(poly), last.(poly), lw=2, linecolor=:white)
end

p_src_sb = plot(; aspect_ratio=:equal,
    xlabel="βx", ylabel="βy",
    title="Source plane: sampled disk + caustic",
    legend=false
)

for poly in caustic_polylines
    plot!(p_src_sb, first.(poly), last.(poly); lw=2)
end

scatter!(p_src_sb, first.(βpts), last.(βpts);
    markersize=1.5, markerstrokewidth=0, alpha=0.25
)

p_overlay_sb = plot(p_sb, p_src_sb; layout=(1,2), size=(1200, 600))

savefig(p_overlay_sb, "generic_cusp_surface_brightness.pdf")
println("Saved: generic_cusp_surface_brightness.pdf")