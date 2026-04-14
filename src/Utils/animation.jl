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
| kwarg                | default      | description                                              |
|----------------------|--------------|----------------------------------------------------------|
| `β_center`           | `(0,0)`      | Source-plane centre of travel path                       |
| `direction`          | `(1.0,-1.0)` | Travel direction (auto-normalised)                       |
| `travel_half_width`  | `0.24`       | Source moves ±this distance from centre                  |
| `n_frames`           | `120`        | Number of animation frames                               |
| `output_file`        | `nothing`    | Path to save GIF/MP4, or `nothing`                       |
| `frame_delay`        | `6`          | ms between GIF frames                                    |
| `fps`                | `30`         | Frames per second for MP4                                |
| `show_source`        | `true`       | Plot unlensed source intensity on the source plane       |
| `show_diff_consec`   | `false`      | Add panel: Δ intensity vs previous frame                 |
| `show_diff_first`    | `false`      | Add panel: Δ intensity vs first frame                    |
"""
function run_animation(lens, src_params, xs_hi, ys_hi,
                       critical_polylines, caustic_polylines,
                       xmin, xmax, ymin, ymax;
                       β_center          = SVector{2,Float64}(0.0, 0.0),
                       direction         = (1.0, -1.0),
                       travel_half_width = 0.24,
                       n_frames          = 120,
                       output_file       = nothing,
                       frame_delay       = 6,
                       fps               = 30,
                       show_source       = true,
                       show_diff_consec  = false,
                       show_diff_first   = false)

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

    # --- determine layout size ---
    n_panels  = 2 + Int(show_diff_consec) + Int(show_diff_first)
    plot_size = (800 * n_panels, 800)

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
 
        # ── collect panels ───────────────────────────────────────────────────
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
            # for poly in critical_polylines
            #     plot!(p_diff_c, first.(poly), last.(poly); lw = 2, linecolor = :black)
            # end
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
            # for poly in critical_polylines
            #     plot!(p_diff_f, first.(poly), last.(poly); lw = 2, linecolor = :black)
            # end
            push!(panels, p_diff_f)
        end
 
        # ── assemble final frame ─────────────────────────────────────────────
        plot(panels...;
            layout        = (1, n_panels),
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
