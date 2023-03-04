#=
This file contains functions that calculate Laguerre and Hermite polynomials.

They are not the fastest way to calculate them for a single input (see ClassicalOrthogonalPolynomials or ApproxFun for that).

Instead, we simply precalculate the polynomial coefficients and then reuse them for parallel calculations, including on GPU.
=#

"""
    binomial(x::Number,y::Number)

Compute the binomial coefficient for noninteger `x` and `y`.
"""
Base.binomial(x::Number, y::Number) = inv((x+1) * beta(x-y+1, y+1))


"""
    laguerre_coefficients(n,α=0;T=Float64)

Compute the coefficients of the nth generalized Laguerre Polynomial.

Converts the coefficients to type float(T).
"""
function laguerre_coefficients(n,α=0;T=Float64)
    ntuple(i->convert(float(T),-(-1)^i*binomial(n+α,n-i+1)/factorial(i-1)),n+1)
end

"""
    laguerre(x,n,α=0)

Compute the the generalized Laguerre Polynomial.

`x` can be a number or an `AbstractArray`.
"""
function laguerre(x,n,α=0)
    c = laguerre_coefficients(n,α,T=eltype(x))
    map(x->evalpoly(x,c),x)
end