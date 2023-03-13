module GPU_propagation

using FreeParaxialPropagation
using FreeParaxialPropagation:reciprocal_grid
using FreeParaxialPropagation:free_propagation_step!
using CUDA,CUDA.CUFFT

function FreeParaxialPropagation.free_propagation(ψ₀::CuArray,xs,ys,z::Number,k,plan,iplan)
    factor = -z/(2k)
    phases = map(ks -> cis(sum(x->factor*x^2,ks)), Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys))) |> collect |> CuArray |> ifftshift

    cache = ifftshift(ψ₀)

    fftshift(free_propagation_step!(cache,phases,plan,iplan))
end

function FreeParaxialPropagation.free_propagation(ψ₀::CuArray,xs,ys,zs::AbstractArray;k=1)
    plan = plan_fft!(ψ₀)
    iplan = plan_ifft!(ψ₀)

    result = similar(ψ₀,size(ψ₀)...,length(zs))

    ks = Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys)) |> collect |> CuArray |> ifftshift

    for (n,z) in enumerate(zs)
        cache = ifftshift(ψ₀)
        factor = -z/(2k)
        phases = map(ks -> cis(sum(x->factor*x^2,ks)), ks)
        free_propagation_step!(cache,phases,plan,iplan)
        fftshift!(view(result,:,:,n),cache)
    end

    result
end

end