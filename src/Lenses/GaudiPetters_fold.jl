# src/Lenses/GauidPetters_fold.jl

using LinearAlgebra
using StaticArrays
using ForwardDiff

struct GaudiPetters_fold <: AbstractLensModel
    a::Float64
    b::Float64
    c::Float64
    d::Float64

    # I am defining an INTERNAL constructor of this struct
    function GaudiPetters_fold(a::Real, b::Real, c::Real, d::Real)
        # Checking the arguments
        if a == 0 || d == 0     # Conditions for fold lens: a and d must be non-zero to avoid degeneracies in the lensing map
            throw(ArgumentError("GaudiPetters_fold requires a and d to be non-zero, got a=$a, d=$d"))
        end

        # Constructing a new instance
        return new(Float64(a), Float64(b), Float64(c), Float64(d))  # calls the struct constructor
    end
end

"""
    deflection(lens::GaudiPetters_fold, θ::SVector{2,Float64}) -> SVector{2,Float64}

Returns the local lensing map for a Gaudi-Petters fold lens - not using the lensing potential or lens equation, but directly implementing the deflection formula from Gaudi & Petters (2002).

Implementation:
- Compute source plane coordinates u1, u2 from image plane coordinates θ1, θ2 using the Gaudi-Petters cusp lens formula.
"""
function deflection(lens::GaudiPetters_fold, θ::SVector{2, Float64}) :: SVector{2, Float64}
    u1 = lens.a * θ[1] + lens.b * θ[2]^2 + lens.c * θ[1] * θ[2]
    u2 = lens.c / 2 * θ[1]^2 + lens.b * θ[1] * θ[2] + lens.d / 2 * θ[2]^2
    return @SVector [u1, u2]
end



"""
    deflection_jacobian(lens::GaudiPetters_fold, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns the jacobian matrix for local lensing map of the GaudiPetters_fold.
"""
function deflection_jacobian(lens::GaudiPetters_fold, θ::SVector{2, Float64}) :: SMatrix{2,2, Float64}
    a, b, c, d = lens.a, lens.b, lens.c, lens.d

    du1_dθ1 = a + c * θ[2]
    du1_dθ2 = c * θ[1] + b * θ[2]
    du2_dθ2 = b * θ[1] + d * θ[2]

    return @SArray [ du1_dθ1 du1_dθ2 ; du1_dθ2 du2_dθ2]
end


