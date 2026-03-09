module Optical_Drift_Effects

# Core deps that many files will want
using LinearAlgebra
using StaticArrays

# -------------------------
# 1) Types + interfaces
# -------------------------
include("Types.jl")

# -------------------------
# 2) Lens models
# -------------------------
include("Lenses/SIS.jl")
include("Lenses/SIE.jl")
include("Lenses/generic_cusp.jl")
include("Lenses/generic_fold.jl")
export generic_cusp, generic_fold, potential, deflection, deflection_jacobian

# export SIS, SIE
# export deflection, deflection_jacobian

# -------------------------
# 3) Source models
# -------------------------
include("Sources/Sersic.jl")
include("Sources/halfPlane.jl")
include("Sources/checkerboard.jl")
export SersicSource, HalfPlaneSource, CheckerboardSource, intensity, sersic_b

# -------------------------
# 3) Public API exports
# -------------------------
# Types
export AbstractLensModel, AbstractSourceModel, AbstractLocalMap
export LensGeometry, GridConfig, SolverConfig, ObservationConfig, RenderConfig, LensingResult

end # module
