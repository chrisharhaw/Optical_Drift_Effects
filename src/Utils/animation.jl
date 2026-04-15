# src/Utils/animation.jl

using Plots
using Printf


"""
    run_animation(lens, src_params, xs_hi, ys_hi,
                  critical_polylines, caustic_polylines,
                  xmin, xmax, ymin, ymax; kwargs...) -> Animation

Animate a source travelling across a caustic by ray-shooting each frame.

# `src_params` — NamedTuple with Sérsic fields
Fields: `I0`, `Re`, `n`, `q`, `ϕ`

# Keyword arguments
| kwarg                    | default      | description                                              |
|--------------------------|--------------|----------------------------------------------------------|
| `β_center`               | `(0,0)`      | Source-plane centre of travel path                       |
| `direction`              | `(1.0,-1.0)` | Travel direction (auto-normalised)                       |
| `travel_half_width`      | `0.24`       | Source moves ±this distance from centre                  |
| `n_frames`               | `120`        | Number of animation frames                               |
| `output_file`            | `nothing`    | Path to save GIF/MP4, or `nothing`                       |
| `frame_delay`            | `6`          | ms between GIF frames                                    |
| `fps`                    | `30`         | Frames per second for MP4                                |
| `show_source`            | `true`       | Plot unlensed source intensity on the source plane       |
| `show_diff_consec`       | `false`      | Add panel: Δ intensity vs previous frame                 |
| `show_diff_first`        | `false`      | Add panel: Δ intensity vs first frame                    |
| `show_obs`               | `false`      | Add second row: pixelated at telescope resolution        |
| `telescope_os`           | `nothing`    | Oversampling factor (integer). Set this when you built   |
|                          |              | xs_hi/ys_hi with `detector_grid` — bypasses calculation  |
| `telescope_pixel_arcsec` | `0.063`      | Detector pixel scale in arcsec; ignored when             |
|                          |              | `telescope_os` is set.                                   |
| `θ_to_arcsec`            | `1.0`        | Conversion: arcsec per θ-unit; ignored when              |
|                          |              | `telescope_os` is set.                                   |
"""
function run_animation(lens, src_params, xs_hi, ys_hi,
                       critical_polylines, caustic_polylines,
                       xmin, xmax, ymin, ymax;
                       β_center              = SVector{2,Float64}(0.0, 0.0),
                       direction             = (1.0, -1.0),
                       travel_half_width     = 0.24,
                       n_frames              = 120,
                       output_file           = nothing,
                       frame_delay           = 6,
                       fps                   = 30,
                       show_source           = true,
                       show_diff_consec      = false,
                       show_diff_first       = false,
                       show_obs              = false,
                       telescope_os          = nothing,
                       telescope_pixel_arcsec = 0.063,
                       θ_to_arcsec           = 1.0)

    # --- build travel path ---
    dx, dy = direction
    norm_d = sqrt(dx^2 + dy^2)
    ux, uy = dx / norm_d, dy / norm_d

    ts     = range(-travel_half_width, travel_half_width; length = n_frames)
    β_path = [SVector{2,Float64}(β_center[1] + ux * t, β_center[2] + uy * t) for t in ts]

    # Pre-extract path arrays (avoids recomputing inside every rendered frame)
    bx = [b[1] for b in β_path]
    by = [b[2] for b in β_path]

    # Helper: build a Sérsic source centred at β0
    function make_source(β0)
        return SersicSource(
            I0        = src_params.I0,
            Re        = src_params.Re,
            n         = src_params.n,
            q         = src_params.q,
            ϕ         = src_params.ϕ,
            β0        = β0,
            normalize = :none,
        )
    end

    # --- pre-compute high-res intensity maps in parallel ---
    println("Ray-shooting $n_frames frames …")
    frames_I = Vector{Matrix{Float64}}(undef, n_frames)
    progress = Threads.Atomic{Int}(0)
    Threads.@threads for k in 1:n_frames
        src         = make_source(β_path[k])
        frames_I[k] = ray_shoot_intensity_map(lens, src, xs_hi, ys_hi)
        p = Threads.atomic_add!(progress, 1) + 1
        p % 10 == 0 && println("  frame $p / $n_frames")
    end

    # Pre-compute log10-clamped intensity maps once — reused for colour scales and rendering
    log_frames_I = [log10.(max.(f, 1e-12)) for f in frames_I]

    # --- pre-compute unlensed source maps on the β grid (no ray-shooting) ---
    frames_src = if show_source
        println("Computing unlensed source maps …"); flush(stdout)
        _src = Vector{Matrix{Float64}}(undef, n_frames)
        Threads.@threads for k in 1:n_frames
            src = make_source(β_path[k])
            # Note: xs_hi/ys_hi are used as the β grid here — same extent as the
            # image plane so the two panels are directly comparable in scale.
            _src[k] = [intensity(src, SVector{2,Float64}(x, y)) for y in ys_hi, x in xs_hi]
        end
        _src
    else
        nothing
    end

    log_frames_src = show_source ? [log10.(max.(f, 1e-12)) for f in frames_src] : nothing

    # Global colour scale for unlensed source panels
    src_clims = if show_source
        (minimum(minimum, log_frames_src), maximum(maximum, log_frames_src))
    else
        nothing
    end

    # --- global colour scale ---
    println("Computing global colour scale …")
    global_min = minimum(minimum, log_frames_I)
    global_max = maximum(maximum, log_frames_I)

    # --- colour scale: consecutive-frame diff ---
    # Δ = log10(I_k) - log10(I_{k-1}), symmetric around zero
    diff_consec_clims = if show_diff_consec
        println("Computing consecutive-frame difference colour scale …"); flush(stdout)
        absmax = 0.0
        for k in 2:n_frames
            for (a, b) in zip(log_frames_I[k], log_frames_I[k-1])
                d = abs(a - b)
                d > absmax && (absmax = d)
            end
        end
        (-absmax, absmax)
    else
        nothing
    end

    # --- colour scale: first-frame diff ---
    # Δ = log10(I_k) - log10(I_1), symmetric around zero
    log_I1_cache = show_diff_first ? log_frames_I[1] : nothing
    diff_first_clims = if show_diff_first
        println("Computing first-frame difference colour scale …"); flush(stdout)
        absmax = 0.0
        for k in 2:n_frames
            for (a, b) in zip(log_frames_I[k], log_I1_cache)
                d = abs(a - b)
                d > absmax && (absmax = d)
            end
        end
        (-absmax, absmax)
    else
        nothing
    end

    # --- telescope-resolution (pixelated) frames ---
    # Downsample linear intensity by block-averaging, then log-transform for display.
    # os = number of hi-res pixels per detector pixel (per axis).
    obs_data = if show_obs
        os = if telescope_os !== nothing
            # Caller passed os directly (e.g. from detector_grid) — use it as-is.
            Int(telescope_os)
        else
            hi_res_step_arcsec = step(xs_hi) * θ_to_arcsec
            os_f = telescope_pixel_arcsec / hi_res_step_arcsec
            os_i = round(Int, os_f)
            if abs(os_f - os_i) > 0.05
                @warn "telescope_pixel_arcsec / hi_res_step = $(round(os_f; digits=2)) is not close " *
                      "to an integer — rounding to os=$(os_i). Use `detector_grid` to build xs_hi/ys_hi " *
                      "from detector parameters, or pass `telescope_os` explicitly."
            end
            os_i
        end
        Nx_hi = length(xs_hi)
        Ny_hi = length(ys_hi)
        if Nx_hi % os != 0 || Ny_hi % os != 0
            error("Grid size ($Ny_hi × $Nx_hi) is not divisible by os=$os. " *
                  "Use `detector_grid` to build a grid that is guaranteed divisible.")
        end
        Nx_obs = Nx_hi ÷ os
        Ny_obs = Ny_hi ÷ os
        xs_obs = range(xmin, xmax; length = Nx_obs)
        ys_obs = range(ymin, ymax; length = Ny_obs)

        println("Computing telescope-resolution frames (os=$os, $(Ny_obs)×$(Nx_obs) pixels) …")
        flush(stdout)
        _obs = Vector{Matrix{Float64}}(undef, n_frames)
        Threads.@threads for k in 1:n_frames
            _obs[k] = block_mean(frames_I[k], os)
        end
        log_obs = [log10.(max.(f, 1e-12)) for f in _obs]
        obs_min = minimum(minimum, log_obs)
        obs_max = maximum(maximum, log_obs)

        # Colour scales for obs-resolution diff panels (mirror hi-res computation)
        obs_diff_consec_clims = if show_diff_consec
            absmax = 0.0
            for k in 2:n_frames
                for (a, b) in zip(log_obs[k], log_obs[k-1])
                    d = abs(a - b)
                    d > absmax && (absmax = d)
                end
            end
            (-absmax, absmax)
        else
            nothing
        end

        log_obs_I1 = show_diff_first ? log_obs[1] : nothing
        obs_diff_first_clims = if show_diff_first
            absmax = 0.0
            for k in 2:n_frames
                for (a, b) in zip(log_obs[k], log_obs_I1)
                    d = abs(a - b)
                    d > absmax && (absmax = d)
                end
            end
            (-absmax, absmax)
        else
            nothing
        end

        (log_obs             = log_obs,
         xs                  = xs_obs,
         ys                  = ys_obs,
         clims               = (obs_min, obs_max),
         os                  = os,
         diff_consec_clims   = obs_diff_consec_clims,
         diff_first_clims    = obs_diff_first_clims,
         log_obs_I1          = log_obs_I1)
    else
        nothing
    end

    # --- determine layout size ---
    n_panels  = 2 + Int(show_diff_consec) + Int(show_diff_first)
    n_rows    = 1 + Int(show_obs)
    plot_size = (800 * n_panels, 800 * n_rows)

    # Pre-extract polyline coordinates to avoid per-frame allocations inside @animate
    critical_xy = [(first.(poly), last.(poly)) for poly in critical_polylines]
    caustic_xy  = [(first.(poly), last.(poly)) for poly in caustic_polylines]

    # --- render animation (sequential — Plots.jl/GR is not thread-safe) ---
    println("Rendering frames …"); flush(stdout)

    anim = @animate for k in 1:n_frames
        print("\r  rendering frame $k / $n_frames"); flush(stdout)
        β_k    = β_path[k]
        log_Ik = log_frames_I[k]

        # ── lens plane ───────────────────────────────────────────────────────
        p_lens = heatmap(xs_hi, ys_hi, log_Ik;
            aspect_ratio  = :equal,
            xlims         = (xmin, xmax),
            ylims         = (ymin, ymax),
            xlabel        = "θx", ylabel = "θy",
            title         = "Lens plane",
            colorbar      = false, legend = false,
            clims         = (global_min, global_max),
        )
        for (px, py) in critical_xy
            plot!(p_lens, px, py; lw = 2, linecolor = :white)
        end

        # ── source plane ─────────────────────────────────────────────────────
        if show_source
            # heatmap of unlensed Sérsic intensity on the β grid
            p_src = heatmap(xs_hi, ys_hi, log_frames_src[k];
                aspect_ratio  = :equal,
                xlims         = (xmin, xmax),
                ylims         = (ymin, ymax),
                xlabel        = "βx", ylabel = "βy",
                title         = "Source plane (unlensed)",
                colorbar      = false, legend = false,
                clims         = src_clims,
            )
            caustic_colour = :white   # caustics visible against dark heatmap
        else
            p_src = plot(;
                aspect_ratio  = :equal,
                xlims         = (xmin, xmax),
                ylims         = (ymin, ymax),
                xlabel        = "βx", ylabel = "βy",
                title         = "Source plane",
                legend        = false,
            )
            caustic_colour = :auto
        end
        for (px, py) in caustic_xy
            plot!(p_src, px, py; lw = 2, linecolor = caustic_colour)
        end
        plot!(p_src, bx, by; lw = 1, linecolor = :gray, linestyle = :dash, alpha = 0.5)
        # scatter!(p_src, [β_k[1]], [β_k[2]]; color = :red, ms = 4, label = "")

        # ── collect row-1 panels ─────────────────────────────────────────────
        panels = Any[p_lens, p_src]

        # consecutive-frame diff panel
        if show_diff_consec
            diff_map = k == 1 ?
                zeros(size(log_Ik)) :
                log_Ik .- log_frames_I[k-1]

            p_diff_c = heatmap(xs_hi, ys_hi, diff_map;
                aspect_ratio  = :equal,
                xlims         = (xmin, xmax),
                ylims         = (ymin, ymax),
                xlabel        = "θx", ylabel = "θy",
                title         = "Δ intensity (vs previous frame)",
                colorbar      = true, legend = false,
                clims         = diff_consec_clims,
                color         = :RdBu,
            )
            push!(panels, p_diff_c)
        end

        # first-frame diff panel
        if show_diff_first
            diff_map_f = k == 1 ?
                zeros(size(log_Ik)) :
                log_Ik .- log_I1_cache

            p_diff_f = heatmap(xs_hi, ys_hi, diff_map_f;
                aspect_ratio  = :equal,
                xlims         = (xmin, xmax),
                ylims         = (ymin, ymax),
                xlabel        = "θx", ylabel = "θy",
                title         = "Δ intensity (vs first frame)",
                colorbar      = true, legend = false,
                clims         = diff_first_clims,
                color         = :RdBu,
            )
            push!(panels, p_diff_f)
        end

        # ── row 2: telescope-resolution panels ───────────────────────────────
        if show_obs
            log_ok = obs_data.log_obs[k]

            # col 1 — pixelated lens plane
            p_obs = heatmap(obs_data.xs, obs_data.ys, log_ok;
                aspect_ratio  = :equal,
                xlims         = (xmin, xmax),
                ylims         = (ymin, ymax),
                xlabel        = "θx", ylabel = "θy",
                title         = "Obs. ($(telescope_pixel_arcsec)\" px⁻¹)",
                colorbar      = false, legend = false,
                clims         = obs_data.clims,
            )
            for (px, py) in critical_xy
                plot!(p_obs, px, py; lw = 1, linecolor = :white, linestyle = :dash)
            end

            # col 2 — blank (no source-plane equivalent at obs resolution)
            obs_row = Any[p_obs, plot(; framestyle = :none)]

            # col 3 — obs consecutive diff
            if show_diff_consec
                obs_diff_map = k == 1 ?
                    zeros(size(log_ok)) :
                    log_ok .- obs_data.log_obs[k-1]

                p_obs_dc = heatmap(obs_data.xs, obs_data.ys, obs_diff_map;
                    aspect_ratio  = :equal,
                    xlims         = (xmin, xmax),
                    ylims         = (ymin, ymax),
                    xlabel        = "θx", ylabel = "θy",
                    title         = "Obs. Δ (vs previous frame)",
                    colorbar      = true, legend = false,
                    clims         = obs_data.diff_consec_clims,
                    color         = :RdBu,
                )
                push!(obs_row, p_obs_dc)
            end

            # col 4 — obs first-frame diff
            if show_diff_first
                obs_diff_f = k == 1 ?
                    zeros(size(log_ok)) :
                    log_ok .- obs_data.log_obs_I1

                p_obs_df = heatmap(obs_data.xs, obs_data.ys, obs_diff_f;
                    aspect_ratio  = :equal,
                    xlims         = (xmin, xmax),
                    ylims         = (ymin, ymax),
                    xlabel        = "θx", ylabel = "θy",
                    title         = "Obs. Δ (vs first frame)",
                    colorbar      = true, legend = false,
                    clims         = obs_data.diff_first_clims,
                    color         = :RdBu,
                )
                push!(obs_row, p_obs_df)
            end

            append!(panels, obs_row)
        end

        # ── assemble final frame ─────────────────────────────────────────────
        plot(panels...;
            layout        = (n_rows, n_panels),
            size          = plot_size,
            left_margin   = 12Plots.mm,
            right_margin  =  6Plots.mm,
            top_margin    =  6Plots.mm,
            bottom_margin = 12Plots.mm,
            plot_title    = @sprintf("frame %03d / %03d   β = (%.4f, %.4f)",
                                      k, n_frames, β_k[1], β_k[2]),
        )
    end

    println("\nFrame count: ", length(anim.frames))

    # --- save ---
    if output_file !== nothing
        ext = lowercase(splitext(output_file)[2])
        if ext == ".gif"
            gif(anim, output_file; fps = round(Int, 1000 / frame_delay))
            println("Saved GIF → $output_file")
        elseif ext == ".mp4"
            mp4(anim, output_file; fps = fps)
            println("Saved MP4 → $output_file")
        else
            @warn "Unknown extension '$ext'; saving as GIF."
            gif(anim, output_file)
        end
    end

    return anim
end
