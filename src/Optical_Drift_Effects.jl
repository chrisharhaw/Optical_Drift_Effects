module Optical_Drift_Effects

# Core deps that many files will want
using LinearAlgebra
using StaticArrays

# -------------------------
# 1) Types + Utils + interfaces
# -------------------------
include("Types.jl")
include("Utils/Mesh.jl")
export block_mean

# -------------------------
# 2) Lens models
# -------------------------
include("Lenses/generic_cusp.jl")
include("Lenses/generic_fold.jl")
include("Lenses/GaudiPetters_cusp.jl")
export generic_cusp, generic_fold, GaudiPetters_cusp, potential, deflection, deflection_jacobian

# -------------------------
# 3) Source models
# -------------------------
include("Sources/Sersic.jl")
include("Sources/halfPlane.jl")
include("Sources/checkerboard.jl")
export SersicSource, HalfPlaneSource, CheckerboardSource, intensity, sersic_b

# -------------------------
# 4) Critical curves and caustics
# -------------------------
include("Caustics/CriticalCurves.jl")
include("Caustics/CausticCurves.jl")
export critical_curves, caustic_curves

# -------------------------
# 5) Observational effects
# -------------------------
include("Observation/PSF.jl")
include("Observation/Detector.jl")
export apply_psf, make_gaussian_psf, add_sky_background, add_poisson_noise, add_read_noise

# -------------------------
# 3) Public API exports
# -------------------------
# Types
export AbstractLensModel, AbstractSourceModel
# export LensGeometry, GridConfig, SolverConfig, ObservationConfig, RenderConfig, LensingResult

end # module
