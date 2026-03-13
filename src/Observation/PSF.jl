# src/Observation/PSF.jl
using DSP
using StaticArrays
using LinearAlgebra

"""
    apply_psf(I, psf_kernel) -> Matrix{Float64}

Convolve intensity map I with psf_kernel using linear (zero-padded) convolution,
then crop back to the original size so the output is centred correctly.

# Arguments
- I : 2D array of intensity values (e.g. from ray-shooting)
- psf_kernel : 2D array representing the PSF kernel (e.g. from `make_gaussian_psf`)

# Returns
2D array of the same size as `I`, representing the intensity map after PSF convolution.
"""
function apply_psf(I::Matrix{Float64}, K::Matrix{Float64})
    Ny, Nx     = size(I)
    Ky, Kx     = size(K)
    half_y     = Ky ÷ 2
    half_x     = Kx ÷ 2
    C          = conv(I, K)                            # DSP.conv: full convolution
    # crop to original size (centred)
    return C[half_y+1 : half_y+Ny, half_x+1 : half_x+Nx]
end

"""
    make_gaussian_psf(σ_θ, pixel_size; truncate=4) -> Matrix{Float64}

Build a normalised 2-D Gaussian PSF kernel.
  σ_θ        : standard deviation in the same angular units as the θ grid
  pixel_size : angular size of one output pixel (xmax-xmin)/(Nx_pix-1)
  truncate   : kernel half-width in units of σ (default 4σ)
The kernel is normalised to sum to 1 so total flux is conserved.
"""
function make_gaussian_psf(σ_θ::Float64, pixel_size::Float64; truncate::Int=4)
    σ_pix = σ_θ / pixel_size                          # σ in pixel units
    half  = ceil(Int, truncate * σ_pix)
    sz    = 2*half + 1
    K     = Matrix{Float64}(undef, sz, sz)
    for j in 1:sz, i in 1:sz
        dx = i - half - 1
        dy = j - half - 1
        K[j, i] = exp(-0.5*(dx^2 + dy^2) / σ_pix^2)
    end
    return K ./ sum(K)
end

