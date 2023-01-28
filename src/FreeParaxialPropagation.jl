module FreeParaxialPropagation

using ClassicalOrthogonalPolynomials:laguerrel,hermiteh
using FFTW, CUDA, CUDA.CUFFT
using Images
using ThreadsX

include("initial_profiles.jl")
export LGConfig,LG,HGConfig,HG,N

include("vizualization.jl")
export vizualize

include("dft_utils.jl")

include("free_propagation.jl")
export free_propagation

end