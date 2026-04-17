# src/Utils/Mesh.jl

# Mesh generation utilities for ray-shooting and intensity map calculations.

# ── Common detector pixel scales (arcsec / pixel) ────────────────────────────
const NIRCAM_LW_PIXEL_ARCSEC  = 0.063   # JWST NIRCam long-wavelength channel
const NIRCAM_SW_PIXEL_ARCSEC  = 0.031   # JWST NIRCam short-wavelength channel
const HST_ACS_WFC_PIXEL_ARCSEC = 0.05   # HST ACS/WFC
const EUCLID_VIS_PIXEL_ARCSEC  = 0.10   # Euclid VIS


"""
    detector_grid(xmin, xmax, ymin, ymax, pixel_arcsec, θ_to_arcsec; os=8)
    -> (xs_hi, ys_hi, xs_pix, ys_pix, os)

Build a hi-res ray-shooting grid that is guaranteed to be an integer multiple
of the detector pixel scale, so `block_mean(..., os)` always works.

# Arguments
- `xmin/xmax/ymin/ymax` — field-of-view extent in θ-units
- `pixel_arcsec`        — detector pixel scale in arcsec (e.g. `NIRCAM_LW_PIXEL_ARCSEC`)
- `θ_to_arcsec`         — conversion factor: arcsec per θ-unit in your simulation
- `os`                  — oversampling factor (hi-res sub-pixels per detector pixel, per axis)

# Returns
- `xs_hi, ys_hi`  — hi-res ray grid  (length = os × Npix per axis)
- `xs_pix, ys_pix` — detector-pixel-centre grid
- `os`            — the oversampling factor (passed through for convenience)

# Example
```julia
xs_hi, ys_hi, xs_pix, ys_pix, os = detector_grid(-2.0, 2.0, -2.0, 2.0,
                                                   NIRCAM_LW_PIXEL_ARCSEC, 0.1;
                                                   os = 8)
```
"""
function detector_grid(xmin, xmax, ymin, ymax,
                       pixel_arcsec::Real, θ_to_arcsec::Real;
                       os::Int = 8)
    pixel_θ  = pixel_arcsec / θ_to_arcsec          # detector pixel size in θ-units
    Nx_pix   = round(Int, (xmax - xmin) / pixel_θ)
    Ny_pix   = round(Int, (ymax - ymin) / pixel_θ)
    xs_hi    = range(xmin, xmax; length = os * Nx_pix)
    ys_hi    = range(ymin, ymax; length = os * Ny_pix)
    xs_pix   = range(xmin, xmax; length = Nx_pix)
    ys_pix   = range(ymin, ymax; length = Ny_pix)
    println("Grid: $(Ny_pix)×$(Nx_pix) detector pixels, " *
            "$(os*Ny_pix)×$(os*Nx_pix) hi-res rays (os=$os)")
    return xs_hi, ys_hi, xs_pix, ys_pix, os
end

""" 
    block_mean(A, os) -> Matrix{Float64}

Downsample a 2D array A by taking the mean of non-overlapping os x os blocks.
# Arguments
- A : 2D array of real numbers to be downsampled
- os : integer block size (output pixel size in input pixel units)
# Returns
- B : 2D array of size (size(A) ÷ os) where each element is the mean of an os x os block from A
# Notes
- The input array A must have dimensions that are divisible by os.

"""
function block_mean(A::AbstractMatrix{<:Real}, os::Int)
    Ny_hi, Nx_hi = size(A)
    @assert Nx_hi % os == 0 "Nx_hi=$(Nx_hi) is not divisible by os=$(os)"
    @assert Ny_hi % os == 0 "Ny_hi=$(Ny_hi) is not divisible by os=$(os)"
    Ny = Ny_hi ÷ os
    Nx = Nx_hi ÷ os
    B = Matrix{Float64}(undef, Ny, Nx)
    inv_os2 = 1.0 / (os * os)

    @inbounds for j in 1:Ny
        j0 = (j-1)*os + 1
        for i in 1:Nx
            i0 = (i-1)*os + 1
            s = 0.0
            for jj in j0:j0+os-1, ii in i0:i0+os-1
                s += A[jj, ii]
            end
            B[j,i] = s * inv_os2
        end
    end
    return B
end