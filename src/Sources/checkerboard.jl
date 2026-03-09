# src/Sources/Checkerboard.jl
using StaticArrays

"""
    CheckerboardSource <: AbstractSourceModel

Axis-aligned checkerboard (grid) surface-brightness pattern on the source plane.
Useful for visualising distortion, shear, and magnification near caustics.

Parameters
- cell_size:   side length of each square cell (same units as β)
- I_hi:        intensity of the "white" squares  (default 1.0)
- I_lo:        intensity of the "black" squares  (default 0.0)
- β0:          origin of the grid in the source plane (βx0, βy0); a grid vertex
               lands exactly on this point
- ϕ:           rotation angle of the grid (radians, CCW).  ϕ=0 gives an axis-
               aligned board.
- window_size: if > 0, the pattern is multiplied by a smooth super-Gaussian
               envelope exp(-(r/window_size)^window_power) so the board fades
               to zero far from β0, which avoids sharp artefacts at the edges of
               your ray-tracing field.  Set to 0.0 to disable.
- window_power: exponent of the envelope (default 6, giving a very flat top).
"""
struct CheckerboardSource <: AbstractSourceModel
    cell_size   ::Float64
    I_hi        ::Float64
    I_lo        ::Float64
    β0          ::SVector{2,Float64}
    ϕ           ::Float64   # grid rotation
    window_size ::Float64   # 0 → no envelope
    window_power::Float64
end

function CheckerboardSource(;
        cell_size   ::Real   = 0.1,
        I_hi        ::Real   = 1.0,
        I_lo        ::Real   = 0.0,
        β0                   = (0.0, 0.0),
        ϕ           ::Real   = 0.0,
        window_size ::Real   = 0.0,
        window_power::Real   = 6.0)

    return CheckerboardSource(
        float(cell_size),
        float(I_hi), float(I_lo),
        SVector{2,Float64}(β0[1], β0[2]),
        float(ϕ),
        float(window_size),
        float(window_power),
    )
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Rotate a 2-vector by angle ϕ (same convention as SersicSource)
@inline function _cb_rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
    c = cos(ϕ); s = sin(ϕ)
    return @SVector [ c*v[1] + s*v[2],
                     -s*v[1] + c*v[2] ]
end

# ---------------------------------------------------------------------------
# Public API  –  same signature as every other source model
# ---------------------------------------------------------------------------

"""
    intensity(src::CheckerboardSource, β::SVector{2,T}) where T<:Number

Returns I_hi or I_lo depending on which checkerboard cell β falls in, optionally
multiplied by a smooth radial envelope.

The cell index (ix, iy) is computed after rotating β into the grid frame.
A cell is "white"  when  (ix + iy) is even, and "black" otherwise.
"""
function intensity(src::CheckerboardSource, β::SVector{2,T}) where {T<:Number}

    # --- 1. shift to grid origin, then rotate into grid frame ---------------
    d  = β - SVector{2,T}(T(src.β0[1]), T(src.β0[2]))
    xy = _cb_rotate(T(src.ϕ), d)       # grid-aligned coordinates

    # --- 2. cell indices (floor division → integer parity) ------------------
    cs  = T(src.cell_size)
    ix  = floor(Int, xy[1] / cs)
    iy  = floor(Int, xy[2] / cs)
    parity = (ix + iy) & 1             # 0 for white, 1 for black

    base = parity == 0 ? T(src.I_hi) : T(src.I_lo)

    # --- 3. optional smooth envelope ----------------------------------------
    ws = T(src.window_size)
    if ws > zero(T)
        r   = sqrt(d[1]*d[1] + d[2]*d[2])
        env = exp(-(r / ws)^T(src.window_power))
        return base * env
    else
        return base
    end
end