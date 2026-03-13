# src/Utils/Mesh.jl

# Mesh generation utilities for ray-shooting and intensity map calculations.




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