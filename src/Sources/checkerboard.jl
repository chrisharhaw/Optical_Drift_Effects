# src/Sources/Checkerboard.jl
using StaticArrays

"""
    CheckerboardSource <: AbstractSourceModel

Axis-aligned checkerboard (grid) surface-brightness pattern on the source plane.
Useful for visualising distortion, shear, and magnification near caustics.

Parameters
- cell_size:      side length of each square cell (same units as β)
- I_hi:           intensity of the "white" squares  (default 1.0)
- I_lo:           intensity of the "black" squares  (default 0.0)
- β0:             origin of the grid in the source plane (βx0, βy0); a grid vertex
                  lands exactly on this point
- ϕ:              rotation angle of the grid (radians, CCW).  ϕ=0 gives an axis-
                  aligned board.
- window_size:    if > 0, pixels outside this radius from β0 return NaN so they
                  render as background in the heatmap.  In standard mode the
                  envelope fades smoothly to zero; in hue_gradient / split_gradient
                  mode a hard mask is applied so the colour encoding is not
                  distorted by a brightness ramp.  Set to 0.0 to disable.
- window_power:   exponent of the smooth envelope in standard mode (default 6).
- hue_gradient:   if true, intensity() returns a value in [0, 1] that encodes
                  x-position as a pure hue ramp, with checker parity applied as
                  a discrete brightness step (dark_fraction).  No smooth envelope
                  is applied so the hue is not corrupted by intensity variation.
                  Use with a cyclic colormap such as :hsv and clims=(0, 1).
                  When false (default), the original scalar I_hi/I_lo behaviour
                  is preserved.
- split_gradient: if true, overrides hue_gradient and uses a split-axis encoding:
                  • parity-0 ("red") tiles → value in [0, 0.5), gradient along y
                  • parity-1 ("blue") tiles → value in [0.5, 1), gradient along x
                  Use with a diverging colormap such as :RdBu and clims=(0, 1).
                  x_range controls the horizontal ramp; y_range the vertical one.
- x_range:        (xmin, xmax) over which the horizontal hue ramps 0 → 1 (source-
                  plane units, applied to the unrotated x-offset from β0).  Points
                  outside are clamped.  Defaults to (-window_size, +window_size)
                  when window_size > 0, else (-1, 1).
- y_range:        (ymin, ymax) over which the vertical hue ramps 0 → 1.  Same
                  default logic as x_range.  Only used in split_gradient mode.
- dark_fraction:  brightness of "black" cells in hue_gradient mode as a fraction
                  of the "white" value (default 0.5).
"""
struct CheckerboardSource <: AbstractSourceModel
    cell_size     ::Float64
    I_hi          ::Float64
    I_lo          ::Float64
    β0            ::SVector{2,Float64}
    ϕ             ::Float64
    window_size   ::Float64
    window_power  ::Float64
    hue_gradient  ::Bool
    split_gradient::Bool          # NEW: per-parity axis split
    x_range       ::SVector{2,Float64}
    y_range       ::SVector{2,Float64}   # NEW: vertical ramp range
    dark_fraction ::Float64
end

function CheckerboardSource(;
        cell_size     ::Real   = 0.1,
        I_hi          ::Real   = 1.0,
        I_lo          ::Real   = 0.0,
        β0                     = (0.0, 0.0),
        ϕ             ::Real   = 0.0,
        window_size   ::Real   = 0.0,
        window_power  ::Real   = 6.0,
        hue_gradient  ::Bool   = false,
        split_gradient::Bool   = false,
        x_range                = nothing,
        y_range                = nothing,
        dark_fraction ::Real   = 0.5)

    half = window_size > 0.0 ? float(window_size) : 1.0

    xr = x_range === nothing ?
        SVector{2,Float64}(-half, half) :
        SVector{2,Float64}(float(x_range[1]), float(x_range[2]))

    yr = y_range === nothing ?
        SVector{2,Float64}(-half, half) :
        SVector{2,Float64}(float(y_range[1]), float(y_range[2]))

    return CheckerboardSource(
        float(cell_size),
        float(I_hi), float(I_lo),
        SVector{2,Float64}(β0[1], β0[2]),
        float(ϕ),
        float(window_size),
        float(window_power),
        hue_gradient,
        split_gradient,
        xr, yr,
        float(dark_fraction),
    )
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@inline function _cb_rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
    c = cos(ϕ); s = sin(ϕ)
    return @SVector [ c*v[1] + s*v[2],
                     -s*v[1] + c*v[2] ]
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    intensity(src::CheckerboardSource, β::SVector{2,T}) where T<:Number

Default (hue_gradient=false, split_gradient=false):
  Returns I_hi or I_lo depending on checker parity, optionally modulated by
  a smooth radial envelope. Returns NaN outside window_size, which renders as
  background in Plots.heatmap.

Hue-gradient mode (hue_gradient=true):
  Returns a value in [0, 1] encoding x-position as a pure hue ramp.
  "White" cells return h(x); "black" cells return h(x) * dark_fraction.
  Points outside window_size return NaN (hard mask, no smooth fade).

Split-gradient mode (split_gradient=true):
  Returns a value in [0, 1] with a per-parity axis:
    • parity-0 tiles → [0.0, 0.5)  driven by y-position (vertical gradient)
    • parity-1 tiles → [0.5, 1.0)  driven by x-position (horizontal gradient)
  Use with a diverging colormap, e.g. color=:RdBu, clims=(0,1).
  Points outside window_size return NaN.
"""
function intensity(src::CheckerboardSource, β::SVector{2,T}) where {T<:Number}

    # --- 1. offset from grid origin (unrotated — used for window and hue) ---
    d = β - SVector{2,T}(T(src.β0[1]), T(src.β0[2]))

    ws = T(src.window_size)

    # --- 2. window handling -------------------------------------------------
    if ws > zero(T)
        r = sqrt(d[1]*d[1] + d[2]*d[2])
        if src.hue_gradient || src.split_gradient
            r > ws && return T(NaN)          # hard mask in colour modes
        else
            env = exp(-(r / ws)^T(src.window_power))
        end
    end

    # --- 3. checker parity (rotated grid frame) -----------------------------
    xy     = _cb_rotate(T(src.ϕ), d)
    cs     = T(src.cell_size)
    ix     = floor(Int, xy[1] / cs)
    iy     = floor(Int, xy[2] / cs)
    parity = (ix + iy) & 1            # 0 = "white/red", 1 = "black/blue"

    # --- 4. split-gradient mode ---------------------------------------------
    #   parity 0 (red tiles) : value in [0, 0.5), gradient along y
    #   parity 1 (blue tiles): value in [0.5, 1), gradient along x
    if src.split_gradient
        if parity == 0
            ymin = T(src.y_range[1]);  ymax = T(src.y_range[2])
            t = clamp((xy[2] - ymin) / (ymax - ymin), zero(T), one(T))
            return t * T(0.5)                # maps [0,1] → [0, 0.5)
        else
            xmin = T(src.x_range[1]);  xmax = T(src.x_range[2])
            t = clamp((xy[1] - xmin) / (xmax - xmin), zero(T), one(T))
            return T(0.5) + t * T(0.5)       # maps [0,1] → [0.5, 1.0)
        end
    end

    # --- 5. hue-gradient mode: pure x-position encoding, no envelope --------
    if src.hue_gradient
        xmin = T(src.x_range[1])
        xmax = T(src.x_range[2])
        h = clamp((xy[1] - xmin) / (xmax - xmin), zero(T), one(T))
        parity_offset = T(src.dark_fraction)
        if parity == 0
            return h * (one(T) - parity_offset)
        else
            return h * (one(T) - parity_offset) + parity_offset
        end
    end

    # --- 6. standard scalar mode --------------------------------------------
    base = parity == 0 ? T(src.I_hi) : T(src.I_lo)
    return ws > zero(T) ? base * env : base
end