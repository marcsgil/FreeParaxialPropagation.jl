function core_LG(x,y,α,γ₀,p,l)
    r2 = (x^2 + y^2)/γ₀^2
    α*exp(-α*r2/2)*(abs(α)*(x+im*sign(l)*y)/γ₀)^abs(l)*laguerrel(p,abs(l),abs2(α)*r2)
end

N_LG(;p,l,k=1,γ₀=1) = 1/(γ₀*√( π*prod(p+1:p+abs(l))))

function LG(x::Number,y::Number,z::Number=0;p=0,l=0,γ₀=1,k=1,normalize=true)
    α = convert(complex(typeof(x)),1/(1+im*z/(k*γ₀^2)))
    
    if normalize
        N(p=p,l=l,k=k,γ₀=γ₀) * cis((2p+abs(l))*angle(α))*core_LG(x,y,α,γ₀,p,l)
    else
        cis((2p+abs(l))*angle(α))*core_LG(x,y,α,γ₀,p,l)
    end
end

function LG(xs::AbstractArray,ys::AbstractArray,z::Number=0;p=0,l=0,γ₀=1,k=1,normalize=true)
    α = convert(complex(eltype(xs)),1/(1+im*z/(k*γ₀^2)))
    prefactor = normalize ? N_LG(p=p,l=l,k=k,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) : cis((2p+abs(l))*angle(α))
    
    ThreadsX.map(r->prefactor*core_LG(r...,α,γ₀,p,l),Iterators.product(xs,ys))
end

function LG(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray;p=0,l=0,γ₀=1,k=1,normalize=true)
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for n in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[n]/(k*γ₀^2)))
        prefactor = normalize ? N_LG(p=p,l=l,k=k,γ₀=γ₀) * cis((2p+abs(l))*angle(α)) : cis((2p+abs(l))*angle(α))
    
        map!(r->prefactor*core_LG(r...,α,γ₀,p,l),view(result,:,:,n),transverse_grid)
    end

    result
end

#Hermite Gaussian

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
end