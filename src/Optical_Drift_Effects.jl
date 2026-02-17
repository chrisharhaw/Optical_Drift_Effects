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

export SIS, SIE
export deflection, deflection_jacobian

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

# make a Sérsic source (defaults are fine too)
src = SersicSource(I0=1.0, Re=0.2, n=1.0, q=0.7, ϕ=0.3, β0=(0.00, 0.00))

# evaluate intensity at a couple of source-plane positions β = (βx, βy)
β1 = @SVector [0.05, -0.02]   # at the center
β2 = @SVector [0.25,  0.10]   # off-center

for r in (0.0, 0.1, 0.2, 0.4, 0.8)
    β = @SVector [r, 0.0]
    println(r, "  ", intensity(src, β))
end

end # module
