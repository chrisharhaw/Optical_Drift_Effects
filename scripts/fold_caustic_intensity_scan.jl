# scripts/fold_caustic_intensity_scan.jl
#
# Compute max|ΔI| per 10-year drift as a function of source position,
# as the source walks from 0.5 Rs outside to 0.5 Rs inside a fold caustic.
#
# Each step = one 10-yr angular displacement Δβ. N_steps is set by the
# path span and Δβ — no subsampling.
#
# To avoid noise from multiple image regions competing, the scan uses only
# the image-plane region around the fold image pair (bottom-left critical arc).
# A single reference pair of full-grid ray-shootings at β_cross locates that
# region; all scan steps ray-shoot within a small zoom window at full resolution.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Optical_Drift_Effects
using LinearAlgebra, StaticArrays, Plots, Printf

# ═══════════════════════════════════════════════════════════════════════════════
# User-adjustable parameters
# ═══════════════════════════════════════════════════════════════════════════════

const HALF_SPAN_RS   = 0.5   # path spans ±this many Rs around the fold caustic
const OS             = 48    # NIRCAM LW oversampling (48 = notebook full precision)
const ZOOM_HW_ARCSEC = 0.4   # ± arcsec around the fold image pair in the image plane
                              # (increase if fold arc drifts outside this window)
const ZOOM_CENTRE = (-0.86087, -0.54203)   # e.g. (θx, θy) in arcsec; or `nothing` for auto-centering on the reference peak
# ═══════════════════════════════════════════════════════════════════════════════

# ── Lens ───────────────────────────────────────────────────────────────────────
σ_v = 229.0;  z_l = 0.227;  z_s = 0.9313;  q_lens = 0.89
lens = SIE_from_vel_disp(σ_v, z_l, z_s, q_lens)
@printf("θ_E = %.4f arcsec\n", lens.θE)

# ── Source effective radius ────────────────────────────────────────────────────
Rs = physical_to_angular_size(0.5, z_s)
@printf("Rs  = %.5f arcsec\n", Rs)

# ── 10-year angular displacement ───────────────────────────────────────────────
ω  = lens_angular_velocity(600.0, z_l)
Δβ = ω * 10.0
@printf("ω   = %.4e arcsec/yr\n", ω)
@printf("Δβ  = %.4e arcsec = %.5f Rs  (10-yr step)\n", Δβ, Δβ / Rs)

# ── NIRCAM-informed full ray-shooting grid ─────────────────────────────────────
xmin, xmax, ymin, ymax = -2.2, 2.2, -2.2, 2.2
xs_full, ys_full, _, _, _ = detector_grid(
    xmin, xmax, ymin, ymax, NIRCAM_LW_PIXEL_ARCSEC, 1.0; os=OS)

# ── Critical and caustic curves ────────────────────────────────────────────────
println("Computing critical/caustic curves …")
xs_cc = range(xmin, xmax; length=10000)
ys_cc = range(ymin, ymax; length=10000)
crits    = critical_curves(lens, xs_cc, ys_cc)
caustics = caustic_curves(lens, crits)

# ── Drift direction: diagonal (1,-1)/√2 ───────────────────────────────────────
drift_dir = normalize(SVector(1.0, -1.0))
drift_dir_perp = SVector(drift_dir[2], -drift_dir[1]) 

# ── Find caustic crossing along diagonal from origin ──────────────────────────
t_cross = let
    best_perp = Inf;  best_t = 0.0
    for poly in caustics
        for pt in poly
            p = SVector(pt[1], pt[2])
            t = dot(p, drift_dir)
            d = norm(p - t * drift_dir)
            if d < best_perp
                best_perp = d
                best_t    = t
            end
        end
    end
    best_t
end

    # const CAUSTIC_CROSS_HINT = SVector(-0.03, 0.03)   

    # t_cross, β_cross_found = let
    #     best_t    = 0.0
    #     best_cross = SVector(0.0, 0.0)
    #     best_dist  = Inf

    #     for poly in caustics
    #         for i in 1:length(poly)-1
    #             a = SVector(poly[i][1],   poly[i][2])
    #             b = SVector(poly[i+1][1], poly[i+1][2])

    #             perp_a = dot(a, drift_dir_perp)
    #             perp_b = dot(b, drift_dir_perp)

    #             # Edge crosses the drift_dir line if perpendicular components change sign
    #             if perp_a * perp_b <= 0
    #                 frac  = perp_a / (perp_a - perp_b)
    #                 cross = a + frac * (b - a)   # actual caustic point at crossing

    #                 # Select the crossing nearest to the hint, not nearest to origin
    #                 dist_to_hint = norm(cross - CAUSTIC_CROSS_HINT)
    #                 if dist_to_hint < best_dist
    #                     best_dist  = dist_to_hint
    #                     best_t     = dot(cross, drift_dir)
    #                     best_cross = cross
    #                 end
    #             end
    #         end
    #     end
    #     best_t, best_cross
    # end

β_cross = t_cross * drift_dir
@printf("Fold crossing: β = (%.5f, %.5f)\n", β_cross[1], β_cross[2])

# ── Path: outside → inside, step = Δβ_10yr ────────────────────────────────────
half_span = HALF_SPAN_RS * Rs
N_steps   = round(Int, 2 * half_span / Δβ) + 1
t_vals    = range(half_span, -half_span; length=N_steps)   # outside → inside
β_path    = [β_cross + t * drift_dir for t in t_vals]

src_at(β0) = SersicSource(I0=1.0, Re=Rs, n=0.5, q=0.7, ϕ=0.0, β0=β0)

# ── Phase 1: reference full-grid computation at β_cross ──────────────────────
# Two ray-shootings at the caustic crossing identify the fold image-pair location.
println("\nPhase 1: reference full-grid computation to locate fold image region …")
I_ref_now  = ray_shoot_intensity_map(lens, src_at(β_cross),                  xs_full, ys_full)
I_ref_10yr = ray_shoot_intensity_map(lens, src_at(β_cross + Δβ * drift_dir), xs_full, ys_full)
abs_ref    = abs.(I_ref_10yr .- I_ref_now)

pk = argmax(abs_ref)
xi_pk, yi_pk = pk[1], pk[2]
    # θx_pk, θy_pk = xs_full[xi_pk], ys_full[yi_pk]
    # @printf("Reference peak |ΔI| = %.3e  at θ = (%.4f, %.4f) arcsec\n",
    #         abs_ref[xi_pk, yi_pk], θx_pk, θy_pk)
θx_auto, θy_auto = xs_full[xi_pk], ys_full[yi_pk]
@printf("Reference peak |ΔI| = %.3e  at θ = (%.4f, %.4f) arcsec  (auto)\n",
        abs_ref[xi_pk, yi_pk], θx_auto, θy_auto)
 
if ZOOM_CENTRE === nothing
    θx_pk, θy_pk = θx_auto, θy_auto
    @printf("Zoom centre: auto (%.4f, %.4f) arcsec\n", θx_pk, θy_pk)
else
    θx_pk, θy_pk = ZOOM_CENTRE
    @printf("Zoom centre: manual (%.4f, %.4f) arcsec  [override]\n", θx_pk, θy_pk)
end


# Reference plot: full image plane with critical curves and zoom-box overlaid
p_ref = heatmap(xs_full, ys_full, log10.(abs_ref .+ 1e-12)';
    aspect_ratio=:equal, colorbar=true, legend=false,
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    xlabel="θx [arcsec]", ylabel="θy [arcsec]",
    title="Reference |ΔI|  (at caustic crossing, full field, os=$OS)")
for poly in crits
    plot!(p_ref, first.(poly), last.(poly); lw=1.5, color=:white, ls=:dash)
end
hw = ZOOM_HW_ARCSEC
plot!(p_ref,
    [θx_pk-hw, θx_pk+hw, θx_pk+hw, θx_pk-hw, θx_pk-hw],
    [θy_pk-hw, θy_pk-hw, θy_pk+hw, θy_pk+hw, θy_pk-hw];
    lw=2, color=:red)
savefig(p_ref, joinpath(@__DIR__, "..", "plots", "fold_scan_reference.png"))
println("Saved → plots/fold_scan_reference.png")

# ── Phase 2: extract zoom window at full resolution ───────────────────────────
xi1 = searchsortedfirst(xs_full, θx_pk - ZOOM_HW_ARCSEC)
xi2 = searchsortedlast( xs_full, θx_pk + ZOOM_HW_ARCSEC)
yi1 = searchsortedfirst(ys_full, θy_pk - ZOOM_HW_ARCSEC)
yi2 = searchsortedlast( ys_full, θy_pk + ZOOM_HW_ARCSEC)
xs_zoom = xs_full[xi1:xi2]
ys_zoom = ys_full[yi1:yi2]

# Pixel area in arcsec² — needed to convert summed intensity to flux units.
# Both axes share the same spacing from the uniform detector_grid.
pixel_area = (xs_zoom[2] - xs_zoom[1]) * (ys_zoom[2] - ys_zoom[1])   # arcsec²
@printf("Pixel area: %.4e arcsec²\n", pixel_area)


n_threads  = Threads.nthreads()
px_zoom    = length(xs_zoom) * length(ys_zoom)
px_full    = length(xs_full) * length(ys_full)
t_est_s    = N_steps * 2 * px_zoom / (n_threads * 6e6)

@printf("\n── Scan summary ─────────────────────────────────────────────────\n")
@printf("  Span:      ±%.5f arcsec = ±%.2f Rs\n",   half_span, HALF_SPAN_RS)
@printf("  Steps:     %d  (10 yr each, %.0f yr total)\n", N_steps, N_steps * 10.0)
@printf("  Full grid: %d × %d px (os=%d)\n",         length(ys_full), length(xs_full), OS)
@printf("  Zoom grid: %d × %d px  (%.3f × %.3f arcsec)\n",
        length(ys_zoom), length(xs_zoom),
        xs_zoom[end]-xs_zoom[1], ys_zoom[end]-ys_zoom[1])
@printf("  Speed-up:  %.0f× vs full grid\n",          px_full / px_zoom)
@printf("  Threads:   %d\n",                           n_threads)
@printf("  Est. time: %.0f s  (≈ %.1f min)\n\n",      t_est_s, t_est_s / 60)

# ── Phase 3: scan within zoom window ─────────────────────────────────────────
    # max_ΔI  = zeros(N_steps)
int_ΔFlux = zeros(N_steps)  
t_start = time()
report_every = max(1, N_steps ÷ 20)

println("Phase 3: scanning $N_steps steps within zoom region …")
for k in 1:N_steps
    I_now  = ray_shoot_intensity_map(lens, src_at(β_path[k]),                  xs_zoom, ys_zoom)
    I_10yr = ray_shoot_intensity_map(lens, src_at(β_path[k] + Δβ * drift_dir), xs_zoom, ys_zoom)

    # Signed integrated flux change over the zoom window.
    # The sum is multiplied by pixel_area to give arcsec² units.
    int_ΔFlux[k] = sum(I_10yr .- I_now) * pixel_area

        #max_ΔI[k] = maximum(abs.(I_10yr .- I_now))

    if k % report_every == 0
        elapsed = time() - t_start
        eta     = elapsed / k * (N_steps - k)
        @printf("  step %5d / %d  |  elapsed %.0f s  |  ETA %.0f s\n",
                k, N_steps, elapsed, eta)
    end
end
@printf("Finished in %.1f s\n", time() - t_start)

abs_int_ΔFlux = abs.(int_ΔFlux)
i_peak = argmax(abs_int_ΔFlux)
@printf("Peak |ΔFlux| = %.4e arcsec²  at %.4f Rs from caustic\n",
        abs_int_ΔFlux[i_peak], t_vals[i_peak] / Rs)

    # i_peak = argmax(max_ΔI)
    # @printf("Peak max|ΔI| = %.4e  at %.4f Rs from caustic\n",
    #         max_ΔI[i_peak], t_vals[i_peak] / Rs)

# ── Result plots ──────────────────────────────────────────────────────────────
pos_Rs = collect(t_vals) ./ Rs   # +HALF_SPAN_RS (outside) … -HALF_SPAN_RS (inside)

# Panel 1: source path on caustic
zoom_src = max(1.5 * Rs, 1.5 * half_span)
p_path = plot(; aspect_ratio=:equal, legend=false,
    xlims=(β_cross[1] - zoom_src, β_cross[1] + zoom_src),
    ylims=(β_cross[2] - zoom_src, β_cross[2] + zoom_src),
    xlabel="βx [arcsec]", ylabel="βy [arcsec]",
    title="Source path through fold caustic")
for poly in caustics
    plot!(p_path, first.(poly), last.(poly); lw=2, color=:black)
end
scatter!(p_path, first.(β_path), last.(β_path);
         ms=2, color=:steelblue, markerstrokewidth=0)
scatter!(p_path, [β_cross[1]], [β_cross[2]]; ms=6, color=:red, markershape=:xcross)

    # # Panel 2: max|ΔI| vs source position
    # p_diff = plot(pos_Rs, max_ΔI; lw=1.5, color=:steelblue, legend=false,
    #     xlabel="Source position relative to fold caustic [Rs]",
    #     ylabel="max |ΔI|  (10-yr baseline)",
    #     title="Max intensity change per 10-yr drift vs source position",
    #     xlims=(-HALF_SPAN_RS, HALF_SPAN_RS) )   # fix: pin axis to the scanned ±HALF_SPAN_RS range
    # vline!(p_diff, [0.0]; color=:red, ls=:dash, lw=1.5)


    # Panel 2: signed integrated ΔFlux vs source position
# Plotting the *signed* curve preserves the physical sense of the caustic crossing
# (net flux rises then falls, or vice versa). The red dashed zero-line makes the
# sign flip at the caustic immediately visible.
p_diff = plot(pos_Rs, int_ΔFlux; lw=1.5, color=:steelblue, legend=false,
    xlabel="Source position relative to fold caustic [Rs]",
    ylabel="Integrated ΔFlux  [arcsec²]  (10-yr baseline)",
    title="Integrated flux change per 10-yr drift vs source position",
    xlims=(-HALF_SPAN_RS, HALF_SPAN_RS))
hline!(p_diff, [0.0]; color=:black, ls=:dot, lw=1.0)   # zero reference
vline!(p_diff, [0.0]; color=:red,   ls=:dash, lw=1.5)  # caustic position


plt = plot(p_path, p_diff;
    layout=(1, 2), size=(1200, 520),
    left_margin=12Plots.mm, right_margin=6Plots.mm,
    top_margin=6Plots.mm, bottom_margin=12Plots.mm)

outpath = joinpath(@__DIR__, "..", "plots", "fold_caustic_intensity_scan_zoom_n2.png")
savefig(plt, outpath)
@printf("Saved → %s\n", outpath)
