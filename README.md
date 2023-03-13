# FreeParaxialPropagation.jl

This is a package that simulates the propagation of a paraxial light beam. Beyond that, it also defines a few important initial profiles, that frequently show up when working with structured light.

It is not yet registered, but you can use it by hitting `]` to enter the Pkg REPL-mode and then typying 

```
add https://github.com/marcsgil/FreeParaxialPropagation.jl.git
```

or by evaluating

```
using Pkg; Pkg.add("https://github.com/marcsgil/FreeParaxialPropagation.jl.git")
```

directyly in a Julia session.

# Example

The following code is a minimal working exemple for this package:

```julia
using FreeParaxialPropagation #Loads the package

rs = LinRange(-5,5,256) #Define a linear grid of points

E0 = lg(rs,rs) #Calculates the fundamental Laguerre-Gaussian mode

vizualize(E0) #Vizualizes the mode

E = free_propagation(E0,rs,rs,1) #Propagates the mode through a distance of 1

vizualize(E) #Vizualizes the evolved mode
```
