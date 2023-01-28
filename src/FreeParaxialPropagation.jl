module FreeParaxialPropagation

using ClassicalOrthogonalPolynomials:laguerrel,hermiteh
using FFTW, CUDA, CUDA.CUFFT
using Images
using ThreadsX

include("initial_profiles.jl")
export LG,HG,diag_HG

include("vizualization.jl")
export vizualize

include("dft_utils.jl")

include("free_propagation.jl")
export free_propagation

end