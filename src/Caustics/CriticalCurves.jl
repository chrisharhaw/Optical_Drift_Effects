# src/CriticalCurves.jl
using StaticArrays
using LinearAlgebra
using Contour

"""
    critical_curves(lens, xs, ys) -> Vector{Vector{SVector{2,Float64}}}

Compute the critical curves of `lens` on the image-plane grid defined by
coordinate vectors `xs` and `ys`.

Critical curves are the loci where the lensing Jacobian is singular,
i.e. where det(∂β/∂θ) = 0. They are returned as a vector of polylines,
each polyline being a vector of 2D image-plane positions.

# Arguments
- `lens`:  any lens object that implements `deflection_jacobian(lens, θ)`
           returning a 2×2 matrix (or SMatrix) at image-plane position θ.
- `xs`:    coordinate vector along the x (first) axis of the image plane.
- `ys`:    coordinate vector along the y (second) axis of the image plane.

# Returns
`Vector{Vector{SVector{2,Float64}}}` — a vector of polylines tracing the
critical curves. Each polyline is a vector of SVector{2,Float64} image-plane
positions. An empty vector is returned if no critical curve is found.

# Notes
The det(J) grid is evaluated at every (x,y) node; the zero contour is then
extracted with `Contour.contours`. Grid resolution therefore directly controls
curve accuracy — use a fine grid (or high oversampling) near cusps.

# Example
- julia
xs = range(-2.0, 2.0; length=1000)
ys = range(-2.0, 2.0; length=1000)
crits = critical_curves(lens, xs, ys)
"""
function critical_curves(lens, xs, ys)
    Nx = length(xs)
    Ny = length(ys)

    detJ = Matrix{Float64}(undef, Nx, Ny)

    @inbounds for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        θ = @SVector [x, y]
        J = deflection_jacobian(lens, θ)
        detJ[i, j] = det(Matrix(J))
    end

    cs = contours(xs, ys, detJ, [0.0])

    polylines = Vector{Vector{SVector{2,Float64}}}()

    for lvl in levels(cs)
        for line in lines(lvl)
            xline, yline = coordinates(line)
            poly = [@SVector [xline[k], yline[k]] for k in eachindex(xline)]
            push!(polylines, poly)
        end
    end

    return polylines
end


"""
    detJ_grid(lens, xs, ys) -> Matrix{Float64}

Return the raw det(∂β/∂θ) grid on `xs × ys`, stored as a (Ny × Nx) matrix
(row = y, column = x, matching the convention of `Plots.heatmap`).

Useful if you need the magnification map or want to pass the grid to your
own contouring / analysis code without recomputing it.
"""
function detJ_grid(lens, xs, ys)
    Nx = length(xs)
    Ny = length(ys)

    detJ = Matrix{Float64}(undef, Ny, Nx)

    @inbounds for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        θ = @SVector [x, y]
        J = deflection_jacobian(lens, θ)
        detJ[j, i] = det(Matrix(J))
    end

    return detJ
end


