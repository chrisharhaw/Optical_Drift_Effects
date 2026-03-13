# src/Observation/Detector.jl

# This file contains functions for simulating the effects of a detector on the observed image:
# - adding a uniform sky background
# - adding Poisson shot noise
# - adding Gaussian read-out noise

using Distributions
using LinearAlgebra
using StaticArrays
using Random

"""
    add_sky_background(I, sky_level) -> Matrix{Float64}

Add a uniform sky background (in the same flux units as I).
"""
add_sky_background(I::Matrix{Float64}, sky::Float64) = I .+ sky

"""
    add_poisson_noise(I, rng) -> Matrix{Float64}

Add Poisson shot noise. I is treated as a photon count map; each pixel is
drawn from Poisson(λ = I[j,i]).  Negative values (from background subtraction)
are clamped to zero before sampling.
"""
function add_poisson_noise(I::Matrix{Float64}, rng::AbstractRNG)
    out = similar(I)
    for k in eachindex(I)
        λ = max(I[k], 0.0)
        out[k] = Float64(rand(rng, Poisson(λ)))
    end
    return out
end

"""
    add_read_noise(I, σ_read, rng) -> Matrix{Float64}

Add Gaussian read-out noise with standard deviation σ_read (in flux units).
"""
function add_read_noise(I::Matrix{Float64}, σ_read::Float64, rng::AbstractRNG)
    return I .+ σ_read .* randn(rng, size(I))
end