

using LinearAlgebra
using StaticArrays
using ForwardDiff

export generic_cusp, deflection, deflection_jacobian

struct generic_cusp <: AbstractLensModel
    d::Int64
    e::Int64
end

function generic_cusp(d::Integer, e::Integer)
    if d != 1 && d != -1
        throw(ArgumentError("generic_cusp requires d=1 or d=-1, got d=$d"))
    end
    if e != 1 && e != -1
        throw(ArgumentError("generic_cusp requires e=1 or e=-1, got e=$e"))
    end
    return generic_cusp(Int64(d), Int64(e))  # calls the struct constructor
end

# --- lensing potential ---------------------------------------------------------
"""
    potential(lens::generic_cusp, θ::SVector{2,Float64}) -> Float64

Returns lensing potential Ψ(θ).

Implementation:
- Compute Ψ in normal coordinates
"""
function potential(lens::generic_cusp, θ::SVector{2,Float64})::Float64
    d = lens.d
    e = lens.e
    x = θ[1]; y = θ[2]
    return (-d/2 * (y^2)* x) + (y^2) /2 - e*(y^4)/4
end


# --- deflection --------------------------------------------------------------

"""
    deflection(lens::generic_cusp, θ::SVector{2,Float64}) -> SVector{2,Float64}

Returns deflection angle α(θ).

Implementation:
- Compute α' in normal coordinates
"""
function deflection(lens::generic_cusp, θ::SVector{2,Float64})::SVector{2,Float64}
    d = lens.d
    e = lens.e
    x = θ[1]; y = θ[2]
    ax = (-d/2) * (y^2)
    ay = (-d * y * x) + y - (e * (y^3))
    return @SVector [ax, ay]
end

"""
    deflection_jacobian(lens::generic_cusp, θ::SVector{2,Float64}) -> SMatrix{2,2,Float64}

Returns ∂α_i/∂θ_j for generic_cusp.
"""
function deflection_jacobian(lens::generic_cusp, θ::SVector{2,Float64})::SMatrix{2,2,Float64}
    d = lens.d
    e = lens.e
    x = θ[1]; y = θ[2]
    J11 = 1.0
    J12 = -d * y
    J21 = -d * y
    J22 = 1 + (d * x) + (3 * e * (y^2))
    return @SMatrix [J11 J12;
                    J21 J22]
end