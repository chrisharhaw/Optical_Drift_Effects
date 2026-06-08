# Optical Drift Effects

In this repository we investigate the effect of tranverse motion of a source object within a strong gravitational lensing system on the resulting images produced and their relative intensities. We ultimately aim to fit the transverse velocity of a source object by the variation of intensity within its associated images.

## Rep Structure

All files required for running the code within this repository are held within the 'src' directory, and imported into the 'Optical_Drift_Effects.jl' script which sets up the package. The list of dependencies for this work are contained within the 'Project.toml' file, simply install through the typical 'using Pkg; Pkg.instantiate()' commands and run scripts within julia as follows:

```
cd Optical_Drift_Effects
julia --project=.
include("path/to/file")
```
WIthin the 'notebooks' directory contains some of our ongoing testing work and withing the 'scripts' directory contains the development of our methods so far. 

## Contact

If you have any questions about the material in this repository, please do not hesitate to contact me at: 

harvey-hawes@cft.edu.pl


