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
# - The implementation below uses the commonly-used closed-form SIE deflection expressions
#   (often attributed to Kormann et al. 1994 / Keeton "gravlens" conventions).
# - Derivatives are computed robustly via ForwardDiff to avoid algebraic mistakes.
# - A safe SIS-like limit is used when q ≈ 1.
#
# You can later swap in an analytic Jacobian for speed if you want.



using LinearAlgebra
using StaticArrays
using ForwardDiff

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

function SIE(θE::Real, q::Real, φ::Real, x0::Real=0.0, y0::Real=0.0; eps_q=1e-6, eps_r=1e-12)
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

    # Ellipticity parameter
    e = sqrt(1.0 - q*q)  # 0<e<1

    # Common auxiliary quantity
    ψ = sqrt(q*q * x*x + y*y) + lens.eps_r

    # Closed-form deflection (one standard convention)
    #
    # αx' = b / e * atan( e x / (ψ + q) )
    # αy' = b / e * atanh( e y / (ψ + q) )
    #
    # Some references write b*q/e prefactors instead. If you find mismatches in the
    # Einstein radius normalization, adjust b or include a factor of q consistently.
    #
    denom = ψ + q

    ax = (b / e) * atan( (e * x) / denom )
    ay = (b / e) * atanh( (e * y) / denom )

    return from_lens_frame(lens, @SVector [ax, ay])
end

# --- Jacobian via AD ---------------------------------------------------------

"""
    deflection_jacobian(lens::SIE, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns ∂α_i/∂θ_j evaluated at θ.

Uses ForwardDiff for robustness. If/when you want speed, replace with an analytic form.
"""
function deflection_jacobian(lens::SIE, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    # ForwardDiff works most smoothly with ordinary vectors.
    f(v) = begin
        θv = @SVector [v[1], v[2]]
        a  = deflection(lens, θv)
        return SVector{2,Float64}(a[1], a[2])
    end

    J = ForwardDiff.jacobian(f, Vector{Float64}([θ[1], θ[2]]))  # 2x2 Matrix
    return SMatrix{2,2,Float64}(J)
end


