#src/Utils/cosmolgy_helper.jl


"""
    _default_cosmology()
 
Returns a flat ΛCDM cosmology matching Planck 2018 best-fit parameters.
Override by passing `cosmo=...` to the public functions below.
"""
_default_cosmology() = cosmology(h=0.6736, OmegaM=0.3153)   # flat ΛCDM, Planck 2018
 
"""
    _dist_Mpc(d)
 
Strip units from a Cosmology.jl distance quantity and return a plain Float64
in Mpc. Cosmology.jl always returns Mpc via UnitfulAstro, so `ustrip(d)`
gives the numeric value directly without needing to specify a target unit.
"""
@inline _dist_Mpc(d) = ustrip(d)   # UnitfulAstro Mpc → plain Float64
 
"""
    angular_diameter_dist_ls(cosmo, z_l, z_s) -> Float64
 
Angular diameter distance from lens plane to source plane [Mpc].
"""
function angular_diameter_dist_ls(cosmo, z_l::Real, z_s::Real)
    return _dist_Mpc(angular_diameter_dist(cosmo, z_l, z_s))
end
 
"""
    angular_diameter_dist_z(cosmo, z) -> Float64
 
Angular diameter distance from observer to redshift z [Mpc].
"""
function angular_diameter_dist_z(cosmo, z::Real)
    return _dist_Mpc(angular_diameter_dist(cosmo, z))
end
 
"""
    physical_to_angular_size(r_phys_kpc, z; cosmo=_default_cosmology()) -> Float64
 
Convert a physical (proper) size `r_phys_kpc` [kpc] at redshift `z` into an
angular size [arcsec] on the sky, using the angular diameter distance D(z).
 
    θ [rad] = r_phys / D(z)
 
where D(z) is the angular diameter distance to redshift z.
 
Use this to set the effective radius `Re` of a `SersicSource` from a physical
half-light radius, so that the source scales correctly with redshift.
 
# Example
```julia
# A galaxy at z=2 with a physical half-light radius of 2 kpc
Re_arcsec = physical_to_angular_size(2.0, 2.0)   # ~ 0.24 arcsec
src = SersicSource(Re=Re_arcsec, ...)
```
 
# Physical intuition
At fixed physical size, higher redshift sources subtend a *smaller* angle
(for z ≳ 1 in ΛCDM, D_A actually turns over and increases again, so very
high-z sources appear larger than their z~1 counterparts — this function
handles that automatically via the cosmology).
"""
function physical_to_angular_size(r_phys_kpc::Real, z::Real;
                                  cosmo=_default_cosmology())
    kpc_per_Mpc = 1.0e3
    D_s_Mpc     = angular_diameter_dist_z(cosmo, z)   # Mpc
    D_s_kpc     = D_s_Mpc * kpc_per_Mpc               # kpc
 
    θ_rad    = r_phys_kpc / D_s_kpc                   # radians
    θ_arcsec = rad2deg(θ_rad) * 3600.0                 # arcsec
    return θ_arcsec
end

 
"""
    lens_angular_velocity(v_trans, z_l; cosmo=_default_cosmology()) -> Float64
 
Convert a lens transverse (peculiar) velocity `v_trans` [km/s] into an
angular drift rate [arcsec / yr] as seen on the sky.
 
The angular velocity of the lens centre is:
 
    dθ/dt = v_trans / (D_l   (1 + z_l))  [radians / s]
 
where D_l is the angular diameter distance to the lens, and the
(1 + z_l) factor accounts for the fact that proper transverse distances
are stretched by (1 + z_l) relative to the comoving ones used for D_l.
 
Returned in arcsec / yr for convenient animation use.
 
# Example
```julia
# A lens at z=0.5 moving at 300 km/s across the line of sight
ω = lens_angular_velocity(300.0, 0.5)   # arcsec/yr
```
"""
function lens_angular_velocity(v_trans::Real, z_l::Real;
                               cosmo=_default_cosmology())
    Mpc_to_km   = 3.085677581e19          # 1 Mpc in km
    # Gyr_to_s    = 3.1557600e16            # 1 Gyr in seconds
    yr_t0_s     = 3.1557600e7             # 1 year in seconds
 
    D_l_Mpc  = angular_diameter_dist_z(cosmo, z_l)   # Mpc
    D_l_km   = D_l_Mpc * Mpc_to_km
 
    # dθ/dt in rad/s
    # Need to be careful with the (1 + z_l) factor here.
    dθ_dt_rad_s = v_trans  / (D_l_km )   # radians/s
 
    # Convert rad/s → arcsec/yr
    dθ_dt_arcsec_yr = rad2deg(dθ_dt_rad_s) * 3600.0 * yr_t0_s
    return dθ_dt_arcsec_yr
end