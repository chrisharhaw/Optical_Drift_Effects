# src/Lenses/SIS.jl
#
# Singular Isothermal Sphere (SIS) lens model.
#
# Provides:
#   - struct SIS
#   - deflection(lens, θ) -> α(θ)
#   - deflection_jacobian(lens, θ) -> ∂α/∂θ
#
# Coordinates are assumed to be in consistent angular units (arcsec or radians).
#
# SIS deflection:
#   α(θ) = θE * θ / |θ|
#
# Jacobian:
#   ∂α/∂θ = θE * ( I/|θ| - θ θᵀ / |θ|^3 )
#
# Numerical safety:
# - eps_r avoids singularity at θ=0.

using LinearAlgebra
using StaticArrays

export SIS, deflection, deflection_jacobian

"""
    SIS(θE, x0, y0; eps_r=1e-12)

Singular Isothermal Sphere lens.

Fields:
- θE : Einstein radius scale
- x0,y0 : lens centre
- eps_r : small radius to avoid division by zero at θ=0
"""
struct SIS <: AbstractLensModel
    θE::Float64
    x0::Float64
    y0::Float64
    eps_r::Float64
end

function SIS(θE::Real, x0::Real=0.0, y0::Real=0.0; eps_r=1e-12)
    return SIS(float(θE), float(x0), float(y0), float(eps_r))
end

@inline function centered(lens::SIS, θ::SVector{2,Float64})
    return θ - @SVector [lens.x0, lens.y0]
end

"""
    deflection(lens::SIS, θ::SVector{2,Float64}) -> SVector{2,Float64}

Returns α(θ) for SIS.
"""
function deflection(lens::SIS, θ::SVector{2,Float64})::SVector{2,Float64}
    d = centered(lens, θ)
    x = d[1]; y = d[2]
    r = hypot(x, y) + lens.eps_r
    a = (lens.θE / r) .* d
    return a
end

"""
    deflection_jacobian(lens::SIS, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns ∂α_i/∂θ_j for SIS.
"""
function deflection_jacobian(lens::SIS, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    d = centered(lens, θ)
    x = d[1]; y = d[2]
    r = hypot(x, y) + lens.eps_r
    r3 = r^3

    I2 = SMatrix{2,2,Float64}((1.0, 0.0,
                              0.0, 1.0))

    outer = @SMatrix [x*x  x*y;
                      x*y  y*y]

    J = lens.θE * (I2 / r - outer / r3)
    return J
end

