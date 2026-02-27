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
export generic_cusp, potential, deflection, deflection_jacobian

# export SIS, SIE
# export deflection, deflection_jacobian

# -------------------------
# 3) Source models
# -------------------------
include("Sources/Sersic.jl")
export SersicSource, intensity, sersic_b

# -------------------------
# 3) Public API exports
# -------------------------
# Types
export AbstractLensModel, AbstractSourceModel, AbstractLocalMap
export LensGeometry, GridConfig, SolverConfig, ObservationConfig, RenderConfig, LensingResult

end # module
