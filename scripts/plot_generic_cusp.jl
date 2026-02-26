# scripts/plot_generic_cusp.jl
using Optical_Drift_Effects
using StaticArrays
using LinearAlgebra
using Plots

# -----------------------------
# USER: construct your lens here
# -----------------------------
# Example:
# lens = GenericCusp( ... )
# or whatever your constructor is

lens = generic_cusp(1, 1)   # <-- change this line to your lens object / parameters

# -----------------------------
# Grid settings
# -----------------------------
Nx, Ny = 300, 300
xmin, xmax = -4.0, 2.0
ymin, ymax = -2.0, 2.0

xs = range(xmin, xmax; length=Nx)
ys = range(ymin, ymax; length=Ny)

# Storage
Ψ   = zeros(Float64, Ny, Nx)   # potential
ax  = zeros(Float64, Ny, Nx)   # deflection x
ay  = zeros(Float64, Ny, Nx)   # deflection y
detJ = zeros(Float64, Ny, Nx)  # det(Jacobian) or det(A); depends on your convention

# -----------------------------
# USER: pointwise evaluation functions
# Replace these wrappers with your actual function names
# -----------------------------
potential_at(lens, θ) = potential(lens, θ)                 # <-- your potential function
deflection_at(lens, θ) = deflection(lens, θ)               # returns SVector(αx, αy)
jacobian_at(lens, θ) = deflection_jacobian(lens, θ)                   # returns 2x2 (SMatrix ok)

# -----------------------------
# Evaluate on grid
# -----------------------------
for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
    θ = @SVector [x, y]

    Ψ[j, i] = potential_at(lens, θ)

    a = deflection_at(lens, θ)
    ax[j, i] = a[1]
    ay[j, i] = a[2]

    J = jacobian_at(lens, θ)
    detJ[j, i] = det(Matrix(J))   # safe if J is SMatrix or Matrix

    betax = x - ax[j, i]
    betay = y - ay[j, i]
    beta = @SVector [betax, betay]
end

# -----------------------------
# Plot: Potential
# -----------------------------
p1 = heatmap(xs, ys, Ψ;
    aspect_ratio=:equal,
    title="Lensing potential Ψ(θ)",
    xlabel="θx", ylabel="θy",
    colorbar_title="Ψ"
)

# -----------------------------
# Plot: Deflection components
# -----------------------------
p2 = heatmap(xs, ys, ax;
    aspect_ratio=:equal,
    title="Deflection ax(θ)",
    xlabel="θx", ylabel="θy",
    colorbar_title="ax"
)

p3 = heatmap(xs, ys, ay;
    aspect_ratio=:equal,
    title="Deflection ay(θ)",
    xlabel="θx", ylabel="θy",
    colorbar_title="ay"
)

# -----------------------------
# Plot: det(J)
# -----------------------------
p4 = heatmap(xs, ys, detJ;
    aspect_ratio=:equal,
    title="det(Jacobian)",
    xlabel="θx", ylabel="θy",
    colorbar_title="detJ"
)

# -----------------------------
# Plot: degnacy curves (det(J)=0) 
# -----------------------------
# You can add contour lines to p4 to show where det(J)=0, which are the critical curves. For example:
contour!(p4, xs, ys, detJ;
    levels=[0.0],  # contour level at det(J)=0
    linewidth=2,
    linecolor=:black,
    label="Critical curve"
)

# -----------------------------
# Caustic curves in source plane 
# -----------------------------



# -----------------------------
# Display / save
# -----------------------------
plot(p1, p2, p3, p4; layout=(2,2), size=(1400, 800))
savefig("generic_cusp_fields.png")
println("Saved: generic_cusp_fields.png")