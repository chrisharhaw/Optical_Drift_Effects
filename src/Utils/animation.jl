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
| kwarg                | default      | description                              |
|----------------------|--------------|------------------------------------------|
| `β_center`           | `(0,0)`      | Source-plane centre of travel path       |
| `direction`          | `(1.0,-1.0)` | Travel direction (auto-normalised)       |
| `travel_half_width`  | `0.24`       | Source moves ±this distance from centre  |
| `n_frames`           | `120`        | Number of animation frames               |
| `output_file`        | `nothing`    | Path to save GIF/MP4, or `nothing`       |
| `frame_delay`        | `6`          | ms between GIF frames                    |
| `fps`                | `30`         | Frames per second for MP4                |
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
                       fps               = 30)

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

    # --- global colour scale (no intermediate array allocations) ---
    println("Computing global colour scale …")
    global_min = Inf
    global_max = -Inf
    for f in frames_I
        for v in f
            lv = log10(max(v, 1e-12))
            lv < global_min && (global_min = lv)
            lv > global_max && (global_max = lv)
        end
    end

    # --- render animation (sequential — Plots.jl/GR is not thread-safe) ---
    println("Rendering frames …"); flush(stdout)
    anim = @animate for k in 1:n_frames
        print("\r  rendering frame $k / $n_frames"); flush(stdout)
        β_k  = β_path[k]
        I_hi = frames_I[k]

        p_lens = heatmap(xs_hi, ys_hi, log10.(I_hi .+ 1e-12);
            aspect_ratio  = :equal,
            xlims         = (xmin, xmax),
            ylims         = (ymin, ymax),
            xlabel        = "θx", ylabel = "θy",
            title         = "Lens plane",
            colorbar      = false, legend = false,
            clims         = (global_min, global_max),
        )
        for poly in critical_polylines
            plot!(p_lens, first.(poly), last.(poly); lw = 2, linecolor = :white)
        end

        p_src = plot(;
            aspect_ratio  = :equal,
            xlims         = (xmin, xmax),
            ylims         = (ymin, ymax),
            xlabel        = "βx", ylabel = "βy",
            title         = "Source plane",
            legend        = false,
        )
        for poly in caustic_polylines
            plot!(p_src, first.(poly), last.(poly); lw = 2)
        end
        plot!(p_src, bx, by; lw = 1, linecolor = :gray, linestyle = :dash, alpha = 0.5)
        scatter!(p_src, [β_k[1]], [β_k[2]]; color = :red, ms = 4, label = "")

        plot(p_lens, p_src;
            layout        = (1, 2),
            size          = (1600, 800),
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
