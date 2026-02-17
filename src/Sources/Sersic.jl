# src/Sources/Sersic.jl
using StaticArrays
using SpecialFunctions   # for gamma / gammainc (can remove if you don't need normalization)

"""
    SersicSource <: AbstractSourceModel

Elliptical Sérsic surface-brightness profile on the source plane.

Parameters
- I0: central intensity (or use Itot with normalize=:total)
- Re: effective (half-light) radius
- n: Sérsic index
- q: axis ratio (0<q<=1), minor/major
- ϕ: position angle (radians), rotation of major axis in the source plane
- β0: center position (βx0, βy0)
- eps_r: small softening to avoid r=0 singularities in some calculations
- normalize: :none (use I0 as given), or :total (use Itot as total flux)
- Itot: total flux if normalize=:total
"""
struct SersicSource <: AbstractSourceModel
    I0::Float64
    Re::Float64
    n::Float64
    q::Float64
    ϕ::Float64
    β0::SVector{2,Float64}
    eps_r::Float64
    normalize::Symbol
    Itot::Float64
end

function SersicSource(; I0::Real=1.0, Re::Real=0.2, n::Real=1.0,
                       q::Real=1.0, ϕ::Real=0.0,
                       β0=(0.0, 0.0),
                       eps_r::Real=1e-12,
                       normalize::Symbol=:none,
                       Itot::Real=1.0)
    return SersicSource(float(I0), float(Re), float(n), float(q), float(ϕ),
                        SVector{2,Float64}(β0[1], β0[2]),
                        float(eps_r), normalize, float(Itot))
end

# --- internal helpers (generic over number type for future AD use) -----------

@inline function _rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
    c = cos(ϕ); s = sin(ϕ)
    return @SVector [ c*v[1] + s*v[2],
                     -s*v[1] + c*v[2] ]
end

@inline function _elliptical_radius(src::SersicSource, β::SVector{2,T}) where {T<:Number}
    d = β - SVector{2,T}(T(src.β0[1]), T(src.β0[2]))
    xy = _rotate(T(src.ϕ), d)  # major axis aligned with x'
    x = xy[1]; y = xy[2]
    q = T(src.q)
    # Common choice: m^2 = x^2 + (y^2)/(q^2)
    return sqrt(x*x + (y*y)/(q*q)) + T(src.eps_r)
end

"""
    sersic_b(n)

Approximate b_n so that Re encloses half the total light.
Good accuracy for n ≳ 0.36.

Common approximation: b_n ≈ 2n - 1/3 + 4/(405n) + 46/(25515 n^2)
"""
@inline function sersic_b(n::T) where {T<:Real}
    return 2n - T(1)/T(3) + T(4)/(T(405)*n) + T(46)/(T(25515)*n*n)
end

# --- public API --------------------------------------------------------------

"""
    intensity(src::SersicSource, β::SVector{2,T}) where T<:Number

Returns surface brightness I(β).
By default uses I0 and returns I0 * exp(-b_n * ((r/Re)^(1/n) - 1)).
If normalize=:total, rescales I0 so that total flux ≈ Itot (elliptical).
"""
function intensity(src::SersicSource, β::SVector{2,T}) where {T<:Number}
    n  = T(src.n)
    Re = T(src.Re)
    r  = _elliptical_radius(src, β)

    bn = sersic_b(n)

    # base profile with I0 parameter interpreted as I_e at r=Re:
    # I(r) = I_e * exp(-b_n * ((r/Re)^(1/n) - 1))
    I_e = T(src.I0)
    I = I_e * exp(-bn * ((r/Re)^(one(T)/n) - one(T)))

    if src.normalize === :none
        return I
    elseif src.normalize === :total
        # Total flux for elliptical Sérsic:
        # F = I_e * 2π q Re^2 n e^{b_n} b_n^{-2n} Γ(2n)
        # (assuming the m definition used above)
        q = T(src.q)
        F = I_e * (2T(pi)) * q * Re*Re * n * exp(bn) * (bn^(-2n)) * T(gamma(2n))
        return I * (T(src.Itot) / F)
    else
        error("Unknown normalize=$(src.normalize). Use :none or :total.")
    end
end
