# src/Lenses/SIE.jl
#
# Singular Isothermal Ellipsoid (SIE) lens model.
#
# This file provides:
#   - struct SIE
#   - deflection(lens, θ) -> α(θ)
#   - deflection_jacobian(lens, θ) -> ∂α/∂θ
#
# Notes:
# - Coordinates are assumed to be in the same angular units everywhere (e.g. arcsec or radians).
# - A safe SIS-like limit is used when q ≈ 1.

using LinearAlgebra
using StaticArrays

export SIE, deflection, deflection_jacobian

"""
    SIE(θE, q, φ, x0, y0; eps_q=1e-6, eps_r=1e-12)

Singular Isothermal Ellipsoid lens.

Fields:
- θE : Einstein radius scale
- q  : axis ratio (0 < q ≤ 1)
- φ  : position angle (radians). Rotation from +x axis to lens major axis.
- x0,y0 : lens centre

Numerical safety:
- eps_q : threshold for treating q≈1 as SIS limit
- eps_r : small radius to avoid division by zero at θ=0
"""
struct SIE <: AbstractLensModel
    θE::Float64
    q::Float64
    φ::Float64
    x0::Float64
    y0::Float64
    eps_q::Float64
    eps_r::Float64
end

function SIE(θE::Real, q::Real, φ::Real=0.0, x0::Real=0.0, y0::Real=0.0; eps_q=1e-6, eps_r=1e-12)
    qf = float(q)
    if !(0.0 < qf <= 1.0)
        throw(ArgumentError("SIE requires 0 < q ≤ 1, got q=$q"))
    end
    return SIE(float(θE), qf, float(φ), float(x0), float(y0), float(eps_q), float(eps_r))
end

# --- small helpers -----------------------------------------------------------

@inline function rotmat(φ::Float64)
    c = cos(φ); s = sin(φ)
    return SMatrix{2,2,Float64}((c, -s,
                                s,  c))
end

@inline function to_lens_frame(lens::SIE, θ::SVector{2,Float64})
    R = rotmat(-lens.φ)
    return R * (θ - @SVector [lens.x0, lens.y0])
end

@inline function from_lens_frame(lens::SIE, v::SVector{2,Float64})
    R = rotmat(lens.φ)
    return R * v
end

# --- deflection --------------------------------------------------------------

"""
    deflection(lens::SIE, θ::SVector{2,Float64}) -> SVector{2,Float64}

Returns deflection angle α(θ).

Implementation:
- Transform θ to lens-aligned frame (major axis along x').
- Compute α' in that frame.
- Rotate back to sky frame.
"""
function deflection(lens::SIE, θ::SVector{2,Float64})::SVector{2,Float64}
    # Lens-frame coordinates (x, y)
    xy = to_lens_frame(lens, θ)
    x = xy[1]; y = xy[2]

    q  = lens.q
    b  = lens.θE

    # If nearly circular, use SIS limit (robust)
    if abs(1.0 - q) < lens.eps_q
        r = hypot(x, y) + lens.eps_r
        ax = b * x / r
        ay = b * y / r
        return from_lens_frame(lens, @SVector [ax, ay])
    end

    denom = sqrt(x^2 + (y/q)^2) 
    ax = b * x / denom
    ay = b * (y/q^2) / denom

    return from_lens_frame(lens, @SVector [ax, ay])
end

# --- Lensing Potential ---------------------------------------------------------
"""
    potential(lens::SIE, θ::SVector{2,Float64}) -> Float64

Returns lensing potential ψ(θ).
The potential is defined such that α = ∇ψ. For SIE, the potential can be computed via a closed-form expression.
"""
function potential(lens::SIE, θ::SVector{2,Float64})::Float64
    # Lens-frame coordinates (x, y)
    xy = to_lens_frame(lens, θ)
    x = xy[1]; y = xy[2]
    q  = lens.q
    b  = lens.θE

    pot = b * sqrt(x^2 + (y/q)^2)
    return pot
end

# --- Jacobian via AD ---------------------------------------------------------
"""
    deflection_jacobian(lens::SIE, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns ∂beta_i/∂θ_j evaluated at θ.

"""
function deflection_jacobian(lens::SIE, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    xy = to_lens_frame(lens, θ)
    x = xy[1]; y = xy[2]
    q  = lens.q
    b  = lens.θE

    J11 = 1 - b * y^2 / (q^2 * (x^2 + (y/q)^2)^(3/2))
    J22 = 1 - b * x^2 / (q^2 * ((x^2 + (y/q)^2)^(3/2)))
    J12 = -b * x * y / (q^2 * (x^2 + (y/q)^2)^(3/2))
    J_lens = @SMatrix [J11  J12;
                       J12  J22]

    return SMatrix{2,2,Float64}(J_lens)
end


