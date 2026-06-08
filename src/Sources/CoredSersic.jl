# src/Sources/CoredSersic.jl
using StaticArrays
using SpecialFunctions   # for gamma (used in :total normalisation)

"""
    CoredSersicSource <: AbstractSourceModel

Elliptical cored-Sérsic surface-brightness profile on the source plane.

The profile is identical to `SersicSource` outside a core radius `r_c`, but
is held flat at `I(r_c)` for all `r < r_c`. This removes the steep central
intensity gradient that arises for high Sérsic indices (n ≳ 1) and suppresses
the discretisation noise those gradients introduce when the source steps across
a pixel grid near a caustic.

Profile:
    r_eff = max(m, r_c)
    I(β)  = I_e * exp(-b_n * ((r_eff / Re)^(1/n) - 1))

where `m` is the usual elliptical radius (see `SersicSource`).

Parameters
- I0:        central intensity at r = Re (i.e. I_e), or total flux if normalize=:total
- Re:        effective (half-light) radius [arcsec]
- n:         Sérsic index
- r_c:       core radius [arcsec]; profile is flat for r < r_c.
             Typical choice: 0.05 Re – 0.2 Re. Set to 0 to recover plain Sérsic.
- q:         axis ratio (0 < q ≤ 1), minor/major
- ϕ:         position angle [radians], rotation of major axis in the source plane
- β0:        centre position (βx0, βy0) [arcsec]
- eps_r:     small softening added to the elliptical radius (avoids r = 0 in
             derivatives; has no visible effect when r_c > 0)
- normalize: :none  → use I0 as I_e directly
             :total → rescale so that total flux equals Itot
- Itot:      target total flux when normalize = :total
"""
struct CoredSersicSource <: AbstractSourceModel
    I0::Float64
    Re::Float64
    n::Float64
    r_c::Float64
    q::Float64
    ϕ::Float64
    β0::SVector{2,Float64}
    eps_r::Float64
    normalize::Symbol
    Itot::Float64
end

function CoredSersicSource(; I0::Real=1.0, Re::Real=0.2, n::Real=1.0,
                             r_c::Real=0.0,
                             q::Real=1.0, ϕ::Real=0.0,
                             β0=(0.0, 0.0),
                             eps_r::Real=1e-12,
                             normalize::Symbol=:none,
                             Itot::Real=1.0)
    r_c >= 0  || throw(ArgumentError("r_c must be ≥ 0"))
    r_c < Re  || @warn "r_c ≥ Re: the entire profile will be flat; " *
                       "consider r_c ∈ [0.05 Re, 0.2 Re]"
    return CoredSersicSource(float(I0), float(Re), float(n), float(r_c),
                             float(q), float(ϕ),
                             SVector{2,Float64}(β0[1], β0[2]),
                             float(eps_r), normalize, float(Itot))
end

# --- internal helpers (generic over number type for future AD use) -----------

@inline function _rotate(ϕ::T, v::SVector{2,T}) where {T<:Number}
    c = cos(ϕ); s = sin(ϕ)
    return @SVector [ c*v[1] + s*v[2],
                     -s*v[1] + c*v[2] ]
end

@inline function _elliptical_radius(src::CoredSersicSource, β::SVector{2,T}) where {T<:Number}
    d  = β - SVector{2,T}(T(src.β0[1]), T(src.β0[2]))
    xy = _rotate(T(src.ϕ), d)
    x  = xy[1]; y = xy[2]
    q  = T(src.q)
    return sqrt(x*x + (y*y)/(q*q)) + T(src.eps_r)
end

# sersic_b is shared with SersicSource — define only if not already defined
@inline function sersic_b(n::T) where {T<:Real}
    return 2n - T(1)/T(3) + T(4)/(T(405)*n) + T(46)/(T(25515)*n*n)
end

# --- public API --------------------------------------------------------------

"""
    intensity(src::CoredSersicSource, β::SVector{2,T}) where T<:Number

Returns surface brightness I(β) for a cored-Sérsic profile.

The elliptical radius `m` is clamped to `max(m, r_c)` before evaluating the
Sérsic exponential, holding the profile flat at `I(r_c)` inside the core.
Normalisation behaviour mirrors `SersicSource`:
- normalize=:none  → I_e = I0 at r = Re (unaffected by r_c)
- normalize=:total → total flux rescaled to Itot (core raises total flux slightly
                     relative to a pure Sérsic; the rescaling accounts for this)
"""
function intensity(src::CoredSersicSource, β::SVector{2,T}) where {T<:Number}
    n   = T(src.n)
    Re  = T(src.Re)
    r_c = T(src.r_c)

    m   = _elliptical_radius(src, β)
    # Clamp to core radius: profile is constant for r < r_c
    r   = max(m, r_c)

    bn  = sersic_b(n)
    I_e = T(src.I0)
    I   = I_e * exp(-bn * ((r/Re)^(one(T)/n) - one(T)))

    if src.normalize === :none
        return I

    elseif src.normalize === :total
        # Total flux for an *uncored* elliptical Sérsic (standard formula):
        #   F_sersic = I_e * 2π q Re² n e^{b_n} b_n^{-2n} Γ(2n)
        #
        # The core adds a filled elliptical disk of radius r_c with intensity
        # I(r_c) = I_e * exp(-b_n * ((r_c/Re)^(1/n) - 1)):
        #   ΔF_core = I(r_c) * π q r_c²   (area of ellipse with semi-axis r_c)
        # minus the Sérsic flux that would have sat inside r_c (which the
        # standard formula already includes via the incomplete gamma integral):
        #   ΔF_sersic_core = I_e * 2π q Re² n e^{b_n} b_n^{-2n}
        #                    * γ(2n, b_n*(r_c/Re)^(1/n))   [lower incomplete Γ]
        # Net correction: F_total = F_sersic - ΔF_sersic_core + ΔF_core
        #
        # For small r_c (the typical use case) this correction is negligible,
        # but it is computed exactly here for correctness.
        q = T(src.q)
        F_sersic = I_e * (2T(pi)) * q * Re*Re * n * exp(bn) *
                   (bn^(-2n)) * T(gamma(2n))

        # lower incomplete gamma: γ(a, x) = gamma(a) * (1 - Γ(a,x)/Γ(a))
        # SpecialFunctions.gamma_inc returns (p, q) where p = γ(a,x)/Γ(a)
        t_c  = bn * (r_c / Re)^(one(T)/n)
        p, _ = gamma_inc(2n, Float64(t_c))           # regularised lower γ
        ΔF_sersic_core = F_sersic * T(p)

        I_at_rc = I_e * exp(-bn * ((r_c/Re)^(one(T)/n) - one(T)))
        ΔF_core = I_at_rc * T(pi) * q * r_c * r_c

        F_total = F_sersic - ΔF_sersic_core + ΔF_core

        return I * (T(src.Itot) / F_total)
    else
        error("Unknown normalize=$(src.normalize). Use :none or :total.")
    end
end