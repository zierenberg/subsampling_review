# subsampling_review
Code that accompanies our Review on methodological advances to make use of subsampled data

The simulation code is using the Julia Language and reproducibility is inspired
by [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).
The plotting code is in Python.


To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This should install all necessary packages for you to be able to run the
scripts and everything should work out of the box, including correctly finding
local paths.
