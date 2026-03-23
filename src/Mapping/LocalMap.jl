#src/Mapping/LocalMap.jl

#Provides the transform from Global lens coordinates (x1, x2) image plane and (y1, y2) source plane to 

using LinearAlgebra
using StaticArrays

# Gaudi & Petters / Congdon, Keeton & Nordgren (2008) local frame
#
# From the Hessian of the lensing potential at an image position, define:
#   a = (1/2) psi_11
#   b =       psi_12
#   c = (1/2) psi_22
#
# The orthogonal matrix M is then:
#
#         1        [ (1-2a)  -b ]
#   M = -----  *  [            ]
#         N        [  b   (1-2a)]
#
# with  N = sqrt((1-2a)^2 + b^2)
#
# We extract a, b, c from deflection_jacobian(lens, theta), which returns
# the matrix A where:
#   A[1,1] = 1 - psi_11,  A[2,2] = 1 - psi_22,  A[1,2] = psi_12


function ckn_abc(lens::SIE, theta::SVector{2,Float64})
    A = deflection_jacobian(lens, theta)
    a = (1.0 - A[1,1]) / 2    # psi_11 / 2
    b = A[1,2]                  # psi_12
    c = (1.0 - A[2,2]) / 2    # psi_22 / 2
    return (a, b, c)
end

function ckn_M(lens::SIE, theta::SVector{2,Float64})
    a, b, _ = ckn_abc(lens, theta)
    p = 1.0 - 2a
    N = sqrt(p^2 + b^2)
    return SMatrix{2,2}(p, b, -b, p) / N   # column-major: col1=(p,b), col2=(-b,p)
end


# --------- Transforms ----------------------------------

# theta = M(theta_ref) * x
function to_local_image(x::AbstractVector, theta_ref::SVector{2,Float64}, lens::SIE)
    M = ckn_M(lens, theta_ref)
    return M * SVector(x[1], x[2])
end

# u = M(theta_ref) * y
function to_local_source(y::AbstractVector, theta_ref::SVector{2,Float64}, lens::SIE)
    M = ckn_M(lens, theta_ref)
    return M * SVector(y[1], y[2])
end

# x = M(theta_ref)' * theta
function from_local_image(theta::AbstractVector, theta_ref::SVector{2,Float64}, lens::SIE)
    M = ckn_M(lens, theta_ref)
    return M' * SVector(theta[1], theta[2])
end

# y = M(theta_ref)' * u
function from_local_source(u::AbstractVector, theta_ref::SVector{2,Float64}, lens::SIE)
    M = ckn_M(lens, theta_ref)
    return M' * SVector(u[1], u[2])
end


# --------- Caustic Classifications ------------------------
# Fold / Cusp classification of caustic/critical curves
#
# At a critical point x_c where det(A) = 0, the caustic velocity is
#   dy/dt = A(x_c) * dx/dt
# Let v = null eigenvector of A, and t = tangent to the critical curve.
#
#   Fold: t NOT parallel to v  -->  dy/dt != 0
#   Cusp: t     parallel to v  -->  dy/dt  = 0
#
# Key insight: the null eigenvector of A in the global frame is exactly the
# second row of M (the degenerate local-frame axis mapped back to global).
# This avoids needing eigen() and sidesteps sign-convention issues in
# deflection_jacobian.

function null_eigenvector(lens::SIE, theta::SVector{2,Float64})
    M = ckn_M(lens, theta)
    return SVector(M[2,1], M[2,2])   # second row of M
end

function classify_critical_point(
        x_prev::AbstractVector, x_curr::AbstractVector, x_next::AbstractVector,
        lens::SIE; thresh::Real = 0.1)
    t = SVector(x_next[1] - x_prev[1], x_next[2] - x_prev[2])
    v = null_eigenvector(lens, SVector(x_curr[1], x_curr[2]))
    sin_angle = abs(t[1]*v[2] - t[2]*v[1]) / (norm(t) * norm(v))
    return sin_angle < thresh ? :cusp : :fold
end

function classify_caustic_polyline(
        crit_poly::AbstractVector, lens::SIE; thresh::Real = 0.1)
    n      = length(crit_poly)
    labels = Vector{Symbol}(undef, n)
    for i in 1:n
        i_prev = mod1(i - 1, n)
        i_next = mod1(i + 1, n)
        labels[i] = classify_critical_point(
            crit_poly[i_prev], crit_poly[i], crit_poly[i_next], lens; thresh)
    end
    return labels
end

"""
    fold_Psi(lens::SIE, theta::SVector{2,Float64})

Returns the third derivatives of the lensing potential transformed via the orthonormal Matrix M into local normal coordinates

"""
function fold_Psi(lens::SIE, theta::SVector{2,Float64})
    M = ckn_M(lens, theta)
    p111, p112, p122, p222 = third_derivatives(lens, theta)
    function psi_ijk(i, j, k)
        s = sort([i, j, k])
        s == [1,1,1] && return p111
        s == [1,1,2] && return p112
        s == [1,2,2] && return p122
                         return p222   # [2,2,2]
    end
    Psi = zeros(2, 2, 2)
    for a in 1:2, b in 1:2, c in 1:2
        v = 0.0
        for i in 1:2, j in 1:2, k in 1:2
            v += M[a,i] * M[b,j] * M[c,k] * psi_ijk(i, j, k)
        end
        Psi[a,b,c] = v
    end
    return Psi
end