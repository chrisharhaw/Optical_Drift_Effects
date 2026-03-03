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
β0 = SVector{2,Float64}(-1.0, 0.4)   # center in source plane 
Rs = 0.05                             # source radius 
N = 60000                              # interior sampling number of points (if used)

src = SersicSource(
    I0=1.0,
    Re=Rs,
    n=2.0, # Sersic index (n=1 exponential, n=4 de Vaucouleurs)
    q=1.0, #axis ratio (1.0 = circular)
    ϕ=0.0, # position angle (radians, for elliptical sources)
    β0=β0,
    normalize=:none 
)

# Choose envelope size for Sersic:
Renvelope = 2 * src.Re

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

# ---------------------------------------------
# Binning for surface density estimation
# ---------------------------------------------
# histogram grid
Nbinsx, Nbinsy = 550, 550
xedges = range(xmin, xmax; length=Nbinsx+1)
yedges = range(ymin, ymax; length=Nbinsy+1)

# bin centers for plotting axes
xcent = @. 0.5*(xedges[1:end-1] + xedges[2:end])
ycent = @. 0.5*(yedges[1:end-1] + yedges[2:end])

# ---------------------------------------------
# lensing functions at a point (wrapper for your lens functions)
# ---------------------------------------------
potential_at(lens, θ) = potential(lens, θ)                 
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)
jacobian_at(lens, θ) = deflection_jacobian(lens, θ)        # returns 2x2 (SMatrix ok)

# ---------------------------------------------
# Evaluate on grid
# ---------------------------------------------
for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
    θ = @SVector [x, y]

    J = jacobian_at(lens, θ)
    detJ[i, j] = det(Matrix(J))   
end


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
# SOURCE PLANE SAMPLING APPROACH (more robust to caustic crossings, but no image curve plotting)
# Note: this approach just plots discrete image points for a grid of source positions
# ---------------------------------------------

"""
    sample_disk(β0, R; N=5000, rng=Random.default_rng())

Uniformly sample N points inside a disk of radius R centered at β0.
Returns Vector{SVector{2,Float64}}.
"""
function sample_disc(β0::SVector{2,Float64}, R::Float64; N::Int=10000, rng=Random.default_rng())
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
     lensed_points(lens, βpts)

 For each β in βpts, compute image positions θ via `image_positions(lens, β)`.
 Returns:
 - θpts::Vector{SVector{2,Float64}} (all image points concatenated)
"""
function lensed_points(lens, βpts::Vector{SVector{2,Float64}})
    θpts = SVector{2,Float64}[]
    for β in βpts
        imgs = image_positions(lens, β)
        append!(θpts, imgs)
    end
    return θpts
end

βpts = sample_disc(β0, Rs; N=N)
θpts = lensed_points(lens, βpts)

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

p_overlay = plot(p_lens, p_src; layout=(1,2), size=(1200, 600))
# savefig(p_overlay, "generic_cusp_extended_cloud.pdf")
# println("Saved: generic_cusp_extended_cloud.pdf")


# ---------------------------------------------
# Binning of points for surface density estimation 
# Note: gives a "pixelated" image that can be used to visualize the surface brightness distribution, but does not give smooth image curves.
# ---------------------------------------------

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

βpts = sample_disc(β0, Rs; N)
θpts = lensed_points(lens, βpts)

H = hist2d(θpts, xedges, yedges)

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

# ---------------------------------------------
# Including non-uniform source brightness (e.g. Sersic profile)
# Note: this requires modifying the sampling to weight points by the source intensity, or applying weights to the histogram counts.
# ---------------------------------------------

"""
Uniformly sample N points in a disk envelope centered at src.β0,
return (βpts, wts) where wts = intensity(src, β).

Renvelope should be large enough to capture the profile wings.
"""
function sample_disk_with_weights(src; N::Int=10000, 
                                Renvelope::Float64=1.0,
                                rng=Random.default_rng())
    β0 = getfield(src, :β0)

    βpts = Vector{SVector{2,Float64}}(undef, N)
    wts  = Vector{Float64}(undef, N)

    for n in 1:N
        r = Renvelope * sqrt(rand(rng))   # uniform in area
        ϕ = 2π * rand(rng)
        β = SVector{2,Float64}(β0[1] + r*cos(ϕ), β0[2] + r*sin(ϕ))
        βpts[n] = β
        wts[n]  = float(intensity(src, β))  #calls from source profile
    end

    return βpts, wts
end


"""
Compute weighted 2D histogram of lensed light.

Adds wts[k] to every image pixel corresponding to βpts[k].
"""
function weighted_lensed_hist2d(lens,
                               βpts::Vector{SVector{2,Float64}},
                               wts::Vector{Float64},
                               xedges::AbstractVector{<:Real},
                               yedges::AbstractVector{<:Real},
                               image_positions_fn)
    nx = length(xedges) - 1
    ny = length(yedges) - 1
    H = zeros(Float64, ny, nx)  # (y,x) order for Plots heatmap

    xmin, xmax = xedges[1], xedges[end]
    ymin, ymax = yedges[1], yedges[end]

    for (β, w) in zip(βpts, wts)
        w == 0.0 && continue
        imgs = image_positions_fn(lens, β)
        for θ in imgs
            x, y = θ[1], θ[2]
            if x < xmin || x >= xmax || y < ymin || y >= ymax
                continue
            end
            ix = clamp(searchsortedlast(xedges, x), 1, nx)
            iy = clamp(searchsortedlast(yedges, y), 1, ny)
            H[iy, ix] += w
        end
    end

    return H
end

βpts, wts = sample_disk_with_weights(src; N=60_000, Renvelope=Renvelope)

H = weighted_lensed_hist2d(lens, βpts, wts, xedges, yedges, image_positions)

Hplot = log10.(H .+ 1e-12)


# --- left panel: lens plane heatmap + critical curve ---
p_lens = heatmap(xcent, ycent, Hplot;
    aspect_ratio=:equal,
    xlabel="θx", ylabel="θy",
    title="Lens plane: lensed Sérsic (weighted) + critical curve",
    colorbar_title="log10(flux+ε)",
    legend=false
)

for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly); lw=2, linecolor=:white)
end

# --- right panel: source plane caustic + sampled source (alpha ∝ weight) ---
p_src = plot(; aspect_ratio=:equal,
    xlabel="βx", ylabel="βy",
    title="Source plane: caustic + sampled Sérsic source",
    legend=false
)

for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly); lw=2)
end

# scale weights to alpha range for visibility
wmax = maximum(wts)
as = [clamp(w / wmax, 0.0, 1.0) for w in wts]

scatter!(p_src, first.(βpts), last.(βpts);
    markersize=1.5,
    markerstrokewidth=0,
    alpha=as
)

p_overlay = plot(p_lens, p_src; layout=(1,2), size=(1200, 600))
savefig(p_overlay, "generic_cusp_sersic_weighted.pdf")
println("Saved: generic_cusp_sersic_weighted.pdf")


# -------------------------------------------------------------
# Animation of source moving across caustic
# -------------------------------------------------------------

function sample_disk_with_weights_relative(src;
        N::Int=30_000,
        Renvelope::Float64=2src.Re,
        rng=Random.default_rng())

    βrel = Vector{SVector{2,Float64}}(undef, N)
    wts  = Vector{Float64}(undef, N)

    # Temporarily treat the source as centered at origin for intensity eval
    src0 = SersicSource(
        I0=src.I0, Re=src.Re, n=src.n, q=src.q, ϕ=src.ϕ,
        β0 = SVector{2,Float64}(0.0, 0.0),
        normalize=src.normalize
    )

    for n in 1:N
        r = Renvelope * sqrt(rand(rng))
        ϕ = 2π * rand(rng)
        p = @SVector [r*cos(ϕ), r*sin(ϕ)]
        βrel[n] = p
        wts[n]  = float(intensity(src0, p))
    end
    return βrel, wts
end

normalize2(v::SVector{2,Float64}) = v / hypot(v[1], v[2])

function line_path(βstart::SVector{2,Float64}, v::SVector{2,Float64}, ts)
    vhat = normalize2(v)
    return [βstart + t*vhat for t in ts]
end

function frame_plot(lens, β0, βrel, wts;
        xedges=xedges, yedges=yedges, xcent=xcent, ycent=ycent,
        critical_polylines=critical_polylines,
        caustic_polylines=caustic_polylines)

    βpts = [β0 + p for p in βrel]

    H = weighted_lensed_hist2d(lens, βpts, wts, xedges, yedges, image_positions)
    Hplot = log10.(H .+ 1e-12)

    p_lens = heatmap(xcent, ycent, Hplot;
        aspect_ratio=:equal,
        xlabel="θx", ylabel="θy",
        title="Lens plane",
        colorbar=false,
        legend=false
    )
    for poly in critical_polylines
        plot!(p_lens, first.(poly), last.(poly); lw=2, linecolor=:white)
    end

    p_src = plot(; aspect_ratio=:equal,
        xlabel="βx", ylabel="βy",
        title="Source plane",
        legend=false
    )
    for poly in caustic_polylines
        plot!(p_src, first.(poly), last.(poly); lw=2)
    end
    scatter!(p_src, [β0[1]], [β0[2]]; markersize=6)

    return plot(p_lens, p_src; layout=(1,2), size=(1200, 600))
end

function animate_crossing(lens;
        outname="cusp_crossing.gif",
        βstart = SVector{2,Float64}(-1.2, 0.3),
        v = SVector{2,Float64}(1.0, 0.0),
        tmin=-0.6,
        tmax=2.0,
        nframes=80,
        Nsrc=8_000,
        rng=MersenneTwister(1))

    βrel, wts = sample_disk_with_weights_relative(src; N=Nsrc, Renvelope=Renvelope, rng=rng)

    ts = range(tmin, tmax; length=nframes)
    path = line_path(βstart, v, ts)

    anim = @animate for β0 in path
        frame_plot(lens, β0, βrel, wts)
    end

    gif(anim, outname; fps=20)
    println("Saved: $outname")
end

# animate_crossing(lens; outname="cross_x.gif",
#     βstart = SVector{2,Float64}(-1.6, 0.2),
#     v      = SVector{2,Float64}(1.0, 0.0))

# animate_crossing(lens; outname="cross_diag.gif",
#     βstart = SVector{2,Float64}(-1.6, -0.4),
#     v      = SVector{2,Float64}(1.0, 0.7))

animate_crossing(lens; outname="cross_vertical.gif",
    βstart = SVector{2,Float64}(-1.0, -1.0),
    v      = SVector{2,Float64}(0.0, 1.0))