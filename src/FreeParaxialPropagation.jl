module FreeParaxialPropagation

using FFTW, CUDA, CUDA.CUFFT
CUDA.allowscalar(false)
using Images
using ThreadsX
using SpecialFunctions

include("orthogonal_polynomials.jl")
export laguerre_coefficients,laguerre

include("initial_profiles.jl")
export lg#,HG,diag_HG

include("vizualization.jl")
export vizualize

include("dft_utils.jl")

include("free_propagation.jl")
export free_propagation

end