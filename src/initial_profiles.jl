parallel_map(f,x...) = first(x) isa CuArray ? map(f,x...) : ThreadsX.map(f,x...)

"""
    binomial(x::Number,y::Number)

Compute the binomial coefficient for noninteger `x` and `y`.
"""
Base.binomial(x::Number, y::Number) = inv((x+1) * beta(x-y+1, y+1))

function build_grid(use_gpu,xs...)
    if use_gpu
        grid = Iterators.product(ntuple(i->Float32.(xs[i]),length(xs))...) |> collect |> CuArray
    else
        grid = Iterators.product(xs...)
    end
    grid
end


function laguerre_coefficients_array(n::Integer,α=0)
    [(-1)^i*binomial(n+α,n-i)/factorial(i) for i in 0:n]
end

"""
    laguerre_coefficients(n::Integer,α=0)

Compute the coefficients of the nth generalized Laguerre Polynomial.
"""
function laguerre_coefficients(n,α=0)
    ntuple(i->-(-1)^i*binomial(n+α,n-i+1)/factorial(i-1),n)
end

normalization_lg(;p,l,γ₀=1) = 1/(γ₀*√( oftype(float(γ₀),π)*prod(p+1:p+abs(l))))

function core_lg(x,y,α,γ₀,l,coefs)
    r2 = (x^2 + y^2)/γ₀^2
    α*exp(-α*r2/2)*(abs(α)*(x+im*sign(l)*y)/γ₀)^abs(l)*evalpoly(abs2(α)*r2,coefs)
end

function _lg(grid,dims::Val{2},z,p,l,γ₀,k,normalize)
    T = eltype(eltype(grid))

    @assert T <: AbstractFloat

    γ₀ = convert(T,γ₀)
    k = convert(T,k)

    coefs = laguerre_coefficients(p,convert(T,abs(l)))

    α = 1/(1+im*z/(k*γ₀^2))
    prefactor = normalize ? normalization_lg(p=p,l=l,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) : cis((2p+abs(l))*angle(α))

    map(r->prefactor*core_lg(r...,α,γ₀,l,coefs), grid)
end

function lg(xs::AbstractArray,ys::AbstractArray,z::Number=0;p::Integer=0,l::Integer=0,γ₀::Real=1,k::Real=1,normalize=true,use_gpu=false)
    grid = build_grid(use_gpu,xs,ys)

    _lg(grid,Val(2),z,p,l,γ₀,k,normalize)
end

function _lg(grid,dims::Val{3},p,l,γ₀,k,normalize)
    T = eltype(eltype(grid))

    @assert T <: AbstractFloat

    γ₀ = convert(T,γ₀)
    k = convert(T,k)

    coefs = laguerre_coefficients(p,convert(T,abs(l)))

    function f(x,y,z)
        α = 1/(1+im*z/(k*γ₀^2))
        prefactor = normalize ? normalization_lg(p=p,l=l,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) : cis((2p+abs(l))*angle(α))
        prefactor*core_lg(x,y,α,γ₀,l,coefs)
    end

    parallel_map(r->f(r...), grid)
end

function lg(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray;p::Integer=0,l::Integer=0,γ₀::Real=1,k::Real=1,normalize=true,use_gpu=false)
    grid = build_grid(use_gpu,xs,ys,zs)

    _lg(grid,Val(3),p,l,γ₀,k,normalize)
end