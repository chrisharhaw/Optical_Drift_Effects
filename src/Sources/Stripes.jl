# src/Sources/Checkerboard.jl
using StaticArrays

"""
StripesSource <: AbstractSourceModel

Surface-brightness pattern on the source plane in the form of stripes, with colour gradient.


Parameters 
- stripe_width:      side length of each square cell (same units as β)
- I_hi:           intensity of the "white" squares  (default 1.0)
- I_lo:           intensity of the "black" squares  (default 0.0)
- β0:             origin of the stripes in the source plane (βx0, βy0)
- ϕ:              rotation angle of the stripes (radians, CCW).  ϕ=0 gives X axis-
                  aligned stripes.
- window_size:    if > 0, pixels outside this radius from β0 return NaN so they
                  render as background in the heatmap.  In standard mode the
                  envelope fades smoothly to zero; in hue_gradient / split_gradient
                  mode a hard mask is applied so the colour encoding is not
                  distorted by a brightness ramp.  Set to 0.0 to disable.
- window_power:   exponent of the smooth envelope in standard mode (default 6).

"""

struct StripesSource <: AbstractSourceModel
    stripe_width    :: Float64
    I_hi            :: Float64
    I_lo            :: Float64
    β0              :: SVector{2, Float64}
    ϕ               :: Float64
    window_size     :: Float64
    window_power    :: Float64
    x_range         :: SVector{2, Float64}
    y_range         :: SVector{2, Float64}
    dark_fraction   :: Float64
end         


function StripesSource(;
    stripe_width    :: Real = 0.1,
    I_hi            :: Real = 1.,
    I_lo            :: Real = 0.,
    β0                      = (0., 0.),
    ϕ               :: Real = 0.,
    window_size     :: Real = 0.0,
    window_power    :: Real = 10.0,
    x_range                 = nothing,
    y_range                 = nothing,
    dark_fraction   :: Real = 0.5
    )

    half = window_size > 0 ? float(window_size) : 1.0

    xr = x_range === nothing ?
        SVector{2,Float64}(-half, half) :
        SVector{2,Float64}(float(x_range[1]), float(x_range[2]))

    yr = y_range === nothing ?
        SVector{2,Float64}(-half, half) :
        SVector{2,Float64}(float(y_range[1]), float(y_range[2]))

    return StripesSource(
        float(stripe_width),
        float(I_hi), float(I_lo),
        SVector{2,Float64}(β0[1], β0[2]),
        float(ϕ),
        float(window_size),
        float(window_power),
        xr, yr,
        float(dark_fraction),
    )
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# @inline function _cb_rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
#     c = cos(ϕ); s = sin(ϕ)
#     return @SVector [ c*v[1] + s*v[2],
#                      -s*v[1] + c*v[2] ]
# end


function intensity(src::StripesSource, β::SVector{2, T}) where T<:Number
    d = β - SVector{2, T}(src.β0[1], src.β0[2])     # move the centres

    ws = T(src.window_size)     # convert to type T
    if ws > zero(T)
        r = sqrt(d[1]^2 + d[2]^2)
        if r > ws return T(NaN) end
    end
    
    xy = _cb_rotate(T(src.ϕ), d)  
    sw = T(src.stripe_width)
   
    parity = floor(Int, xy[2] / sw) & 1      # floor(Int, x / sw) & 1 could be a faster alternative

    return parity == 1 ? 1. : src.dark_fraction 
end

