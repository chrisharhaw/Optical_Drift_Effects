# src/Lenses/generic_fold.jl

using LinearAlgebra
using StaticArrays
using ForwardDiff

export generic_fold, deflection, deflection_jacobian

struct generic_fold <: AbstractLensModel
    d::Int64
end

function generic_fold(d::Integer)
    if d != 1 && d != -1
        throw(ArgumentError("generic_fold requires d=1 or d=-1, got d=$d"))
    end
    return generic_fold(Int64(d))  # calls the struct constructor
end

# --- lensing potential ---------------------------------------------------------
"""
    potential(lens::generic_fold, θ::SVector{2,Float64}) -> Float64

Returns lensing potential Ψ(θ).

Implementation:
- Compute Ψ in normal coordinates
"""
function potential(lens::generic_fold, θ::SVector{2,Float64})::Float64
    d = lens.d
    x = θ[1]; y = θ[2]
    return (-d/3 * (y^3)) + (y^2) /2 
end

# --- deflection --------------------------------------------------------------

"""
    deflection(lens::generic_fold, θ::SVector{2,Float64}) -> SVector{2,Float64}

Returns deflection angle α(θ).

Implementation:
- Compute α' in normal coordinates
"""
function deflection(lens::generic_fold, θ::SVector{2,Float64})::SVector{2,Float64}
    d = lens.d
    x = θ[1]; y = θ[2]
    ax = 0.0
    ay = (-d * (y^2)) + y
    return @SVector [ax, ay]
end

# --- deflection jacobian --------------------------------------------------------------

"""
    deflection_jacobian(lens::generic_fold, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns the Jacobian matrix of the deflection angle, i.e., the magnification matrix A(θ) = I - ∇α(θ).
Implementation:
- Compute the Jacobian of the deflection angle using ForwardDiff
"""
function deflection_jacobian(lens::generic_fold, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    d = lens.d
    x = θ[1]; y = θ[2]

    J11 = 1.0
    J12 = 0.0
    J21 = 0.0
    J22 = 2 * d * y
    return @SMatrix [J11 J12;
                    J21 J22]
end

