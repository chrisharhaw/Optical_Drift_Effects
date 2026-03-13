# src/Lenses/GaudiPetters_cusp.jl

using LinearAlgebra
using StaticArrays
using ForwardDiff

struct GaudiPetters_cusp <: AbstractLensModel
    a::Float64
    b::Float64
    c::Float64
end

function GaudiPetters_cusp(a::Real, b::Real, c::Real)
    if b == 0 || c == 0 || (2a*c - b^2) == 0
        throw(ArgumentError("GaudiPetters_cusp requires b, c, and (2ac-b^2) to be nonzero, got b=$b, c=$c, 2ac-b^2=$(2a*c - b^2)"))
    end
    return GaudiPetters_cusp(Float64(a), Float64(b), Float64(c))  # calls the struct constructor
end

# --- deflection --------------------------------------------------------------

"""
    deflection(lens::GaudiPetters_cusp, θ::SVector{2,Float64}) -> SVector{2,Float64}

Returns the local lensing map for a Gaudi-Petters cusp lens - not using the lensing potential or lens equation, but directly implementing the deflection formula from Gaudi & Petters (2002).

Implementation:
- Compute source plane coordinates u1, u2 from image plane coordinates θ1, θ2 using the Gaudi-Petters cusp lens formula.
"""

function deflection(lens::GaudiPetters_cusp, θ::SVector{2,Float64})::SVector{2,Float64}
    a, b, c= lens.a, lens.b, lens.c
    x = θ[1]; y = θ[2]
    u1 = c*x + (b/2)*y^2
    u2 = b*x*y + a*y^3 
    return @SVector [u1, u2]
end

"""
    deflection_jacobian(lens::GaudiPetters_cusp, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns the jacobian matrix for local lensing map of the GaudiPetters_cusp.
"""
function deflection_jacobian(lens::GaudiPetters_cusp, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    a, b, c = lens.a, lens.b, lens.c
    x = θ[1]; y = θ[2]
    du1_dx = c
    du1_dy = b*y
    du2_dx = b*y
    du2_dy = b*x + 3*a*y^2
    return @SMatrix [du1_dx du1_dy; 
                    du2_dx du2_dy]
end

