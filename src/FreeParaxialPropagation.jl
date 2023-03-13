module FreeParaxialPropagation

using Reexport

using FFTW
using Images, ImageShow
@reexport using ColorSchemes
using ThreadsX
using SpecialFunctions

include("initial_profiles.jl")
export lg,hg,diagonal_hg

include("vizualization.jl")
export vizualize,animate

include("dft_utils.jl")

include("free_propagation.jl")
export free_propagation

end