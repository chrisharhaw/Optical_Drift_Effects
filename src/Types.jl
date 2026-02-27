# src/Types.jl
#
# Core data types used across the package.
#
# Philosophy:
# - Put data containers here: configuration structs, model containers,
#   and result containers that many modules will share.
# - Don't put algorithms here (solvers, rendering, contouring) 

#THERE IS A LOT IN HERE RN THAT I WON'T BE USING YET, BUT IT'S A starting point for the data structures I expect to need.
# We can trim it down as we go if it's too much.

using StaticArrays

# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

abstract type AbstractLensModel end
abstract type AbstractSourceModel end

# Optional: later you can add AbstractLocalMap for near-caustic work
abstract type AbstractLocalMap end

# ---------------------------------------------------------------------------
# Geometry / cosmology container (optional for now)
# ---------------------------------------------------------------------------

"""
    LensGeometry(zL, zS)

Container for lens/source redshifts and (later) distance factors.
Keep minimal at first; expand once you need Σcrit and physical units.
"""
struct LensGeometry
    zL::Float64
    zS::Float64
end

LensGeometry(zL::Real, zS::Real) = LensGeometry(float(zL), float(zS))

# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------

"""
    GridConfig(Nx, Ny, xmin, xmax, ymin, ymax)

Defines a uniform grid in the image plane or source plane.
Units are whatever your lens/source use (arcsec recommended).
"""
struct GridConfig
    Nx::Int
    Ny::Int
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

function GridConfig(Nx::Integer, Ny::Integer,
                    xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    Nx < 2 && throw(ArgumentError("Nx must be ≥ 2"))
    Ny < 2 && throw(ArgumentError("Ny must be ≥ 2"))
    xmax <= xmin && throw(ArgumentError("xmax must be > xmin"))
    ymax <= ymin && throw(ArgumentError("ymax must be > ymin"))
    return GridConfig(Int(Nx), Int(Ny), float(xmin), float(xmax), float(ymin), float(ymax))
end

# Convenience: square grid constructor
GridConfig(N::Integer, xmin::Real, xmax::Real, ymin::Real, ymax::Real) =
    GridConfig(N, N, xmin, xmax, ymin, ymax)

# ---------------------------------------------------------------------------
# Solver configuration (for point image finding, contour refinement, etc.)
# ---------------------------------------------------------------------------

"""
    SolverConfig(; method=:newton, tol=1e-10, maxiter=50, nseeds=64)

Generic solver settings used by root finders, multistart searches, etc.
"""
Base.@kwdef struct SolverConfig
    method::Symbol = :newton
    tol::Float64 = 1e-10
    maxiter::Int = 50
    nseeds::Int = 64
end

# ---------------------------------------------------------------------------
# Observation / instrument configuration
# ---------------------------------------------------------------------------

"""
    ObservationConfig(; psf_fwhm, pixel_scale, read_noise, sky_level, exposure_time)

Simple instrument model:
- psf_fwhm: PSF FWHM in same angular units as grids (e.g. arcsec)
- pixel_scale: pixel size (arcsec/pixel)
- read_noise: Gaussian read noise (e-)
- sky_level: sky background level (e-/pixel or counts/pixel; pick a convention)
- exposure_time: seconds (or arbitrary scaling if you're not modelling throughput yet)
"""
Base.@kwdef struct ObservationConfig
    psf_fwhm::Float64 = 0.7
    pixel_scale::Float64 = 0.2
    read_noise::Float64 = 5.0
    sky_level::Float64 = 50.0
    exposure_time::Float64 = 100.0
end

# ---------------------------------------------------------------------------
# Rendering configuration (optional but helpful)
# ---------------------------------------------------------------------------

"""
    RenderConfig(; oversample=1, roi_pad=0.0)

- oversample: render on a finer grid then bin down (helps PSF/pixel integration)
- roi_pad: optional padding around ROI when rendering (arcsec)
"""
Base.@kwdef struct RenderConfig
    oversample::Int = 1
    roi_pad::Float64 = 0.0
end

# ---------------------------------------------------------------------------
# A generic result container (you can move this into Pipeline/Results.jl if preferred)
# ---------------------------------------------------------------------------

"""
    LensingResult

A lightweight container for common outputs.
Keep fields optional so early prototypes don't need everything filled.
"""
Base.@kwdef mutable struct LensingResult
    # Critical curves in image plane: vector of polylines, each polyline is a vector of 2-vectors
    critical_curves::Union{Nothing, Vector{Vector{SVector{2,Float64}}}} = nothing

    # Caustic curves in source plane
    caustic_curves::Union{Nothing, Vector{Vector{SVector{2,Float64}}}} = nothing

    # Point-source image positions
    point_images::Union{Nothing, Vector{SVector{2,Float64}}} = nothing

    # Point-source diagnostics (μ, parity, etc.)
    point_diagnostics::Any = nothing

    # Extended lensed image (pre-instrument)
    lensed_image::Union{Nothing, Matrix{Float64}} = nothing

    # Observed image (PSF/pixel/noise applied)
    observed_image::Union{Nothing, Matrix{Float64}} = nothing

    # Optional diagnostic maps
    detA_map::Union{Nothing, Matrix{Float64}} = nothing
    mu_map::Union{Nothing, Matrix{Float64}} = nothing
end
