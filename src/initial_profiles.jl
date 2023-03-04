function core_lg(x,y,α,γ₀,l,coefs)
    r2 = (x^2 + y^2)/γ₀^2
    α*exp(-α*r2/2)*(abs(α)*(x+im*sign(l)*y)/γ₀)^abs(l)*evalpoly(abs2(α)*r2,coefs)
end

normalization_lg(;p,l,γ₀=1) = 1/(γ₀*√( π*prod(p+1:p+abs(l))))

function lg(x::Real,y::Real,z::Real=0;p=0,l=0,γ₀=1,k=1,normalize=true)
    α = oftype(complex(x),1/(1+im*z/(k*γ₀^2)))

    coefs = laguerre_coefficients(p,abs(l);T=typeof(x))
    
    if normalize
        normalization_lg(p=p,l=l,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) * core_lg(x,y,α,γ₀,l,coefs)
    else
        cis((2p+abs(l))*angle(α)) * core_lg(x,y,α,γ₀,l,coefs)
    end
end

function lg(xs::AbstractArray,ys::AbstractArray,z::Number=0;p=0,l=0,γ₀=1,k=1,normalize=true,use_gpu=false)

    if use_gpu
        iterator = Iterators.product(Float32.(xs),Float32.(ys)) |> collect |> CuArray
    else
        iterator = Iterators.product(xs,ys)
    end

    T = eltype(eltype(iterator))

    α = convert(complex(T),1/(1+im*z/(k*γ₀^2)))
    prefactor = normalize ? normalization_lg(p=p,l=l,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) : cis((2p+abs(l))*angle(α))
    coefs = laguerre_coefficients(p,abs(l);T=T)

    map(r->prefactor*core_lg(r[1],r[2],α,γ₀,l,coefs), iterator)
end

#=function lg(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray;p=0,l=0,γ₀=1,k=1,normalize=true)
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for n in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[n]/(k*γ₀^2)))
        prefactor = normalize ? normalization_lg(p=p,l=l,k=k,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) : cis((2p+abs(l))*angle(α))
    
        map!(r->prefactor*core_LG(r...,α,γ₀,p,l),view(result,:,:,n),transverse_grid)
    end

    result
end=#

#=Hermite Gaussian

function core_HG(x,y,α,γ₀,m,n)
    ξ = x/γ₀
    η = y/γ₀
    α*exp(-α*(ξ^2+η^2)/2)*hermiteh(m,abs(α)*ξ)*hermiteh(n,abs(α)*η)
end

N_HG(;m,n,k=1,γ₀=1) = 1/(γ₀*√( π*2^((m+n))*factorial(n)*factorial(m)))

function HG(xs::AbstractArray,ys::AbstractArray,z::Number=0;m=0,n=0,γ₀=1,k=1,normalize=true)
    α = convert(complex(eltype(xs)),1/(1+im*z/(k*γ₀^2)))
    prefactor = normalize ? N_HG(m=m,n=n,k=k,γ₀=γ₀) * cis((m+n)*angle(α)) : cis((m+n)*angle(α))
    
    ThreadsX.map(r->prefactor*core_HG(r...,α,γ₀,m,n),Iterators.product(xs,ys))
end

function HG(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray;m=0,n=0,γ₀=1,k=1,normalize=true)
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for i in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[i]/(k*γ₀^2)))
        prefactor = normalize ? N_HG(m=m,n=n,k=k,γ₀=γ₀) * cis((m+n)*angle(α)) : cis((m+n)*angle(α))
    
        map!(r->prefactor*core_HG(r...,α,γ₀,m,n),view(result,:,:,i),transverse_grid)
    end

    result
end

## Diaganal HG

function core_diag_HG(x,y,α,γ₀,m,n)
    ξ = (x+y)/(√2γ₀)
    η = (x-y)/(√2γ₀)
    α*exp(-α*(ξ^2+η^2)/2)*hermiteh(m,abs(α)*ξ)*hermiteh(n,abs(α)*η)
end

function diag_HG(xs::AbstractArray,ys::AbstractArray,z::Number=0;m=0,n=0,γ₀=1,k=1,normalize=true)
    α = convert(complex(eltype(xs)),1/(1+im*z/(k*γ₀^2)))
    prefactor = normalize ? N_HG(m=m,n=n,k=k,γ₀=γ₀) * cis((m+n)*angle(α)) : cis((m+n)*angle(α))
    
    ThreadsX.map(r->prefactor*core_diag_HG(r...,α,γ₀,m,n),Iterators.product(xs,ys))
end

function diag_HG(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray;m=0,n=0,γ₀=1,k=1,normalize=true)
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for i in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[i]/(k*γ₀^2)))
        prefactor = normalize ? N_HG(m=m,n=n,k=k,γ₀=γ₀) * cis((m+n)*angle(α)) : cis((m+n)*angle(α))
    
        map!(r->prefactor*core_diag_HG(r...,α,γ₀,m,n),view(result,:,:,i),transverse_grid)
    end

    result
end=#