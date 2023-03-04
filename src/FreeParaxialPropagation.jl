module FreeParaxialPropagation

using Reexport

using FFTW, CUDA, CUDA.CUFFT
CUDA.allowscalar(false)
using Images, ImageShow
@reexport using ColorSchemes
using ThreadsX
using SpecialFunctions

include("initial_profiles.jl")
export build_grid,laguerre_coefficients,lg#,HG,diag_HG

include("vizualization.jl")
export vizualize,make_animation

include("dft_utils.jl")

include("free_propagation.jl")
export free_propagation

end