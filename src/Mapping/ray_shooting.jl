#src/Mapping/ray_shooting.jl



"""
    ray_shoot_intensity_map(lens, src, xs, ys) -> Matrix{Float64}

Compute the image-plane intensity map I(θ) on a grid by ray-shooting.
Each pixel maps θ → β = θ - α(θ), then evaluates the source intensity at β.
"""
function ray_shoot_intensity_map(lens, src, xs, ys)
    Nx = length(xs)
    Ny = length(ys)
    I  = Matrix{Float64}(undef, Ny, Nx)
    @inbounds for j in 1:Ny
        y = ys[j]
        for i in 1:Nx
            θ       = SVector{2,Float64}(xs[i], y)
            β       = θ - deflection(lens, θ)
            I[i, j] = Float64(intensity(src, β))
        end
    end
    return I
end