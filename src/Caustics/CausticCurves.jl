# src/CausticCurves.jl
using StaticArrays


"""
    caustic_curves(lens, critical_polylines) -> Vector{Vector{SVector{2,Float64}}}

Map a set of critical-curve polylines from the image plane to the source plane
to obtain the caustic curves.

Each critical-curve point θ is mapped to the source plane via the lens equation
    β = θ - α(θ)
where α is the deflection angle returned by `deflection(lens, θ)`.

# Arguments
- `lens`:               any lens object implementing `deflection(lens, θ)`,
                        returning an SVector{2} deflection at image position θ.
- `critical_polylines`: output of `critical_curves(lens, xs, ys)` — a
                        `Vector{Vector{SVector{2,Float64}}}` of image-plane
                        polylines.

# Returns
`Vector{Vector{SVector{2,Float64}}}` — one source-plane polyline per input
critical-curve polyline, in the same order.

# Example
- julia
xs = range(-2.0, 2.0; length=1000)
ys = range(-2.0, 2.0; length=1000)

crits   = critical_curves(lens, xs, ys)
caustics = caustic_curves(lens, crits)

"""
function caustic_curves(lens, critical_polylines)
    caustic_polylines = Vector{Vector{SVector{2,Float64}}}()

    for poly in critical_polylines
        mapped = [θ - deflection(lens, θ) for θ in poly]
        push!(caustic_polylines, mapped)
    end

    return caustic_polylines
end
