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
using Cosmology

export SIE, deflection, deflection_jacobian
export sie_einstein_radius, SIE_from_vel_disp

# ============================================================================
# Einstein radius from physical parameters
# ============================================================================
 
"""
    sie_einstein_radius(σ_v, z_l, z_s; cosmo=_default_cosmology()) -> Float64
 
Compute the SIE Einstein radius θ_E in **arcseconds** from:
 
  - `σ_v`  : velocity dispersion of the lens in km/s
  - `z_l`  : lens redshift
  - `z_s`  : source redshift
  - `cosmo`: a Cosmology.jl cosmology object (default: flat ΛCDM, Planck 2018)
 
Formula (SIS / SIE):
 
    θ_E = 4π (σ_v/c)² · D_ls / D_s     [radians]
 
where D_ls and D_s are angular diameter distances.
 
References: Kochanek (2006), Bartelmann & Schneider (2001).
"""
function sie_einstein_radius(σ_v::Real, z_l::Real, z_s::Real;
                             cosmo=_default_cosmology())
    z_l >= z_s && throw(ArgumentError(
        "Lens redshift z_l=$z_l must be less than source redshift z_s=$z_s"))
 
    c_km_s = 2.99792458e5   # speed of light in km/s
 
    D_s  = angular_diameter_dist_z(cosmo, z_s)    # Mpc
    D_ls = angular_diameter_dist_ls(cosmo, z_l, z_s)  # Mpc
 
    # θ_E in radians
    θE_rad = 4π * (σ_v / c_km_s)^2 * (D_ls / D_s)
 
    # Convert to arcseconds
    θE_arcsec = rad2deg(θE_rad) * 3600.0
    return θE_arcsec
end

# ============================================================================
# Convenience constructor: build SIE from physical parameters
# ============================================================================
 
"""
    SIE_from_vel_disp(σ_v, z_l, z_s, q, φ=0.0, x0=0.0, y0=0.0;
                      cosmo=_default_cosmology(),
                      eps_q=1e-6, eps_r=1e-12) -> SIE
 
Construct an SIE lens from physical parameters.  The Einstein radius is
computed from angular diameter distances via `sie_einstein_radius`, and all
spatial coordinates (x0, y0) are assumed to be in the same arcsecond units
as θ_E.
 
Arguments:
  - `σ_v`  : lens velocity dispersion [km/s]
  - `z_l`  : lens redshift
  - `z_s`  : source redshift
  - `q`    : axis ratio (0 < q ≤ 1)
  - `φ`    : position angle [radians], default 0.0
  - `x0,y0`: lens centre position [arcsec], default (0, 0)
  - `cosmo`: Cosmology.jl cosmology (default: flat ΛCDM Planck 2018)
 
Returns an SIE struct with θE set in arcseconds.
 
# Example
```julia
cosmo = cosmology(h=0.70, OmegaM=0.30)   # custom cosmology
lens  = SIE_from_vel_disp(230.0, 0.5, 2.0, 0.8; cosmo=cosmo)
println("θ_E = \$(lens.θE) arcsec")
```
"""
function SIE_from_vel_disp(σ_v::Real, z_l::Real, z_s::Real,
                           q::Real, φ::Real=0.0,
                           x0::Real=0.0, y0::Real=0.0;
                           cosmo=_default_cosmology(),
                           eps_q=1e-6, eps_r=1e-12)
    θE = sie_einstein_radius(σ_v, z_l, z_s; cosmo=cosmo)
    return SIE(θE, q, φ, x0, y0; eps_q=eps_q, eps_r=eps_r)
end

# ============================================================================
# SIE lens model
# ============================================================================

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
    # function deflection(lens::SIE, θ::SVector{2,Float64})::SVector{2,Float64}
    #     # Lens-frame coordinates (x, y)
    #     xy = to_lens_frame(lens, θ)
    #     x = xy[1]; y = xy[2]

    #     q  = lens.q
    #     b  = lens.θE

    #     # If nearly circular, use SIS limit (robust)
    #     if abs(1.0 - q) < lens.eps_q
    #         r = hypot(x, y) + lens.eps_r
    #         ax = b * x / r
    #         ay = b * y / r
    #         return from_lens_frame(lens, @SVector [ax, ay])
    #     end

    #     denom = sqrt(x^2 + (y/q)^2) 
    #     ax = b * x / denom
    #     ay = b * (y/q^2) / denom

    #     return from_lens_frame(lens, @SVector [ax, ay])
    # end

function deflection(lens::SIE, θ::SVector{2,Float64})::SVector{2,Float64}
    xy = to_lens_frame(lens, θ)
    x = xy[1]; y = xy[2]
    q = lens.q
    b = lens.θE

    if abs(1.0 - q) < lens.eps_q
        r = hypot(x, y) + lens.eps_r
        return from_lens_frame(lens, @SVector [b * x / r, b * y / r])
    end

    sq  = sqrt(1.0 - q^2)
    xi  = sqrt((q*x)^2 + y^2) + lens.eps_r   # elliptical radius

    ax = (b * q / sq) * atan(x * sq / xi)
    ay = (b * q / sq) * atanh(y * sq  / xi)   # https://arxiv.org/pdf/astro-ph/0102341

    return from_lens_frame(lens, @SVector [ax, ay])
end

# --- Lensing Potential ---------------------------------------------------------
"""
    potential(lens::SIE, θ::SVector{2,Float64}) -> Float64

Returns lensing potential ψ(θ).
The potential is defined such that α = ∇ψ. For SIE, the potential can be computed via a closed-form expression.
"""
    # function potential(lens::SIE, θ::SVector{2,Float64})::Float64
    #     # Lens-frame coordinates (x, y)
    #     xy = to_lens_frame(lens, θ)
    #     x = xy[1]; y = xy[2]
    #     q  = lens.q
    #     b  = lens.θE

    #     pot = b * sqrt(x^2 + (y/q)^2)
    #     return pot
    # end


# --- Jacobian via AD ---------------------------------------------------------
"""
    deflection_jacobian(lens::SIE, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns ∂beta_i/∂θ_j evaluated at θ.

"""
    # function deflection_jacobian(lens::SIE, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    #     xy = to_lens_frame(lens, θ)
    #     x = xy[1]; y = xy[2]
    #     q  = lens.q
    #     b  = lens.θE

    #     J11 = 1 - b * y^2 / (q^2 * (x^2 + (y/q)^2)^(3/2))
    #     J22 = 1 - b * x^2 / (q^2 * ((x^2 + (y/q)^2)^(3/2)))
    #     J12 = -b * x * y / (q^2 * (x^2 + (y/q)^2)^(3/2))
    #     J_lens = @SMatrix [J11  J12;
    #                     J12  J22]

    #     return SMatrix{2,2,Float64}(J_lens)
    # end

function deflection_jacobian(lens::SIE, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    xy = to_lens_frame(lens, θ)
    x = xy[1]; y = xy[2]
    q  = lens.q
    b  = lens.θE

    xi  = sqrt((q*x)^2 + y^2) + lens.eps_r

    J11 = 1 - b*q*y^2 / ((x^2 + y^2) * xi)
    J22 = 1 - b*q*x^2 / ((x^2 + y^2) * xi)
    J12 = -b * q * x * y / ((x^2 + y^2) * xi)
    J_lens = @SMatrix [J11  J12;
                    J12  J22]

    return SMatrix{2,2,Float64}(J_lens)
end

# --- Third Derivatives ------------------------------------------------------
"""
    third_derivatives(lens::SIE, theta::SVector{2,Float64})

Returns the third derivatives of the lensing potential.

"""

    # function third_derivatives(lens::SIE, theta::SVector{2,Float64})
    #     # Work in lens frame (handles phi != 0 correctly)
    #     x1, x2 = lens.x0 == 0.0 && lens.y0 == 0.0 && lens.φ == 0.0 ?
    #             (theta[1], theta[2]) :
    #             let xy = SVector(cos(-lens.φ)*(theta[1]-lens.x0) - sin(-lens.φ)*(theta[2]-lens.y0),
    #                             sin(-lens.φ)*(theta[1]-lens.x0) + cos(-lens.φ)*(theta[2]-lens.y0))
    #                 (xy[1], xy[2])
    #             end
    #     b_sie = lens.θE
    #     q     = lens.q
    #     xi    = sqrt(x1^2 + x2^2 / q^2)
    #     fac   = b_sie / (q^2 * xi^5)
    #     q2xi2 = q^2 * xi^2
    #     psi111 = -3fac * x1 * x2^2
    #     psi112 =  fac * x2 * (2q2xi2 - 3x2^2) / q^2
    #     psi122 =  fac * x1 * (3x2^2 - q2xi2)  / q^2
    #     psi222 = -3fac * x1^2 * x2             / q^2
    #     return (psi111, psi112, psi122, psi222)
    # end

function third_derivatives(lens::SIE, theta::SVector{2,Float64})
    # Work in lens frame (handles phi != 0 correctly)
    x, y = lens.x0 == 0.0 && lens.y0 == 0.0 && lens.φ == 0.0 ?
            (theta[1], theta[2]) :
            let xy = SVector(cos(-lens.φ)*(theta[1]-lens.x0) - sin(-lens.φ)*(theta[2]-lens.y0),
                            sin(-lens.φ)*(theta[1]-lens.x0) + cos(-lens.φ)*(theta[2]-lens.y0))
                (xy[1], xy[2])
            end
    b_sie = lens.θE
    q     = lens.q

    denom = (x^2 + y^2)^2 * (q^2 * x^2 + y^2)^(3/2)
    prefac = -b_sie * q / denom
  
    psi111 = prefac * y^2 * x * (3q^2 * x^2 + (q^2 +2) * y^2)
    psi112 =  prefac * y * (-y^4 + x^2 * y^2 + 2q^2 * x^4)
    psi122 =  prefac * x * (-q^2 * x^4 + q^2 * x^2 * y^2 + 2y^4)
    psi222 = prefac * x^2 * y * (3y^2 + (2q^2 + 1) * x^2)
    return (psi111, psi112, psi122, psi222)
end


