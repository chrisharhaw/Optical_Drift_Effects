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

# -------------------------
# 3) Public API exports
# -------------------------
# Types
export AbstractLensModel, AbstractSourceModel, AbstractLocalMap
export LensGeometry, GridConfig, SolverConfig, ObservationConfig, RenderConfig, LensingResult

# Lenses
export SIS, SIE
export deflection, deflection_jacobian

end # module
