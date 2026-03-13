# scripts/halfPlane_ray_shooting.jl
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

# lens = generic_fold(1)   

# ------------------------------
# Source set-up
# ------------------------------
normalize2(v::SVector{2,Float64}) = v / hypot(v[1], v[2])

βstar = SVector{2,Float64}(0.0, 0.0)              # point on boundary line
n     = normalize2(SVector{2,Float64}(0.3, 1.0))  # normal toward bright side
tangent_from_normal(n::SVector{2,Float64}) = SVector{2,Float64}(-n[2], n[1])
w     = 0.02                                      # steepness scale in β-units 

src = HalfPlaneSource(1.0, 0.0, βstar, n, w)

# -----------------------------
# Grid settings
# -----------------------------

os = 20          # oversampling factor (pixels per output pixel, per axis)

Nx_pix, Ny_pix = 50, 50                 # desired output (pixelated) resolution
Nx_hi,  Ny_hi  = os * Nx_pix, os * Ny_pix  # high-res ray grid

xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0

# High-res ray grid 
xs_hi = range(xmin, xmax; length=Nx_hi)
ys_hi = range(ymin, ymax; length=Ny_hi)

# Output pixel centres (for plot axes)
xs_pix = range(xmin, xmax; length=Nx_pix)
ys_pix = range(ymin, ymax; length=Ny_pix)

# ---------------------------------------------
# Observation parameters  <-- adjust these
# ---------------------------------------------
psf_σ_θ    = 0.04     # PSF Gaussian σ in θ-units (same units as xmin/xmax)
sky_level  = 0.005    # uniform sky background in flux units
flux_scale = 2e4     # scale factor: converts normalised intensity → photon counts
σ_read     = 5.0     # read noise standard deviation in photon counts
obs_seed   = 42      # RNG seed for reproducibility

pixel_size = (xmax - xmin) / (Nx_pix - 1)   # angular size of one output pixel


# ---------------------------------------------
# lensing functions at a point (wrapper for your lens functions)
# ---------------------------------------------             
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)


# ---------------------------------------------
#  Critical curves: find det(J)=0 contours in image plane
# ---------------------------------------------
critical_polylines = critical_curves(lens, xs_hi, ys_hi)
caustic_polylines = caustic_curves(lens, critical_polylines)

# --------------------------------------------
# Image positions for a specific source position β
# ---------------------------------------------
# THIS IS NOT USED IN THE RAY-SHOOTING APPROACH BELOW WHICH IS GNERAL FOR EXTENDED SOURCES.
# This method for finding image positions is specific to cusp lens models and could be useful for point source or to check ray-tracing results.
# As well as this, it could be used for tracking specific points in the source plane as they move across the caustic, and defining the magnification for a point source.
# A different method would be needed for a general lens model or a fold caustic, but the ray-shooting approach below is general and works for any lens and extended source.


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

# --- draw half-plane in the source plane: boundary + shaded bright side ---
function add_halfplane_to_sourceplot!(p, βstar::SVector{2,Float64}, n::SVector{2,Float64};
        L::Float64=5.0, α::Float64=0.25)

    n_hat = normalize2(n)
    tdir = tangent_from_normal(n_hat)

    # boundary line (for display)
    tt = range(-L, L; length=200)
    xs = [ (βstar + τ*tdir)[1] for τ in tt ]
    ys = [ (βstar + τ*tdir)[2] for τ in tt ]
    plot!(p, xs, ys; lw=2, linestyle=:dash)

    # shaded bright side: big rectangle extending along +n_hat
    a  = βstar - L*tdir
    b  = βstar + L*tdir
    a2 = a + L*n_hat
    b2 = b + L*n_hat

    polyx = [a[1], b[1], b2[1], a2[1]]
    polyy = [a[2], b[2], b2[2], a2[2]]
    plot!(p, polyx, polyy; seriestype=:shape, opacity=α, linealpha=0.0)

    return p
end

#---generate ray-shooting intensity map and plot with caustics/critical curves ---

I_hi  = ray_shoot_intensity_map(lens, src, xs_hi, ys_hi)
I_pix = block_mean(I_hi, os)

# ---------------------------------------------
# Observational effects
# ---------------------------------------------
psf_kernel = make_gaussian_psf(psf_σ_θ, pixel_size)
println("PSF kernel size: $(size(psf_kernel)), σ = $(psf_σ_θ) θ-units = $(psf_σ_θ/pixel_size) pixels")

I_psf  = apply_psf(I_pix, psf_kernel)                         # 1. PSF convolution
I_sky  = add_sky_background(I_psf, sky_level)                 # 2. sky background
I_cnt  = I_sky .* flux_scale                                   # 3. scale to counts

rng    = MersenneTwister(obs_seed)
I_shot = add_poisson_noise(I_cnt,  rng)                       # 4. Poisson shot noise
I_obs  = add_read_noise(I_shot, σ_read, rng)                  # 5. Gaussian read noise

# --------------------------------------------
# Plotting
# --------------------------------------------

# log-stretch for display (guard against ≤0 after noise)
log_stretch(A) = log10.(max.(A, 1.0))

Iplot = log10.(I_pix .+ 1e-12)

p_lens_hi = heatmap(xs_hi, ys_hi, log10.(I_hi .+ 1e-12);  # log for display
    aspect_ratio=:equal,
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    xlabel="θx", ylabel="θy",
    title="Lens plane: high-res",
    colorbar=false, legend=false
)

for poly in critical_polylines
    plot!(p_lens_hi, first.(poly), last.(poly); lw=2, linecolor=:white)
end

p_lens = heatmap(xs_pix, ys_pix, Iplot;
    aspect_ratio=:equal,
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    xlabel="θx", ylabel="θy",
    title="Lens plane: pixelated",
    colorbar=false, legend=false
)

for poly in critical_polylines
    plot!(p_lens, first.(poly), last.(poly); lw=2, linecolor=:white)
end
println("Plotted lens plane intensity map.")

p_obs = heatmap(xs_pix, ys_pix, log_stretch(I_obs);
    aspect_ratio=:equal,
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    xlabel="θx", ylabel="θy",
    title="Observed (PSF + noise)",
    colorbar=false, legend=false
)
for poly in critical_polylines
    plot!(p_obs, first.(poly), last.(poly); lw=2, linecolor=:white)
end

p_src = plot(; aspect_ratio=:equal,
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    xlabel="βx", ylabel="βy",
    title="Source plane",
    legend=false
)

for poly in caustic_polylines
    plot!(p_src, first.(poly), last.(poly); lw=2)
end
println("Plotted caustic curves in source plane.")

add_halfplane_to_sourceplot!(p_src, βstar, n; L=5.0, α=0.25)

p_overlay = plot(p_lens_hi, p_lens, p_obs, p_src; layout=(2,2), size=(1400, 1400),  left_margin=12Plots.mm, right_margin=6Plots.mm,
    top_margin=6Plots.mm,   bottom_margin=12Plots.mm)

savefig(p_overlay, "halfplane_fold.png")
println("Saved: halfplane_fold.png")

