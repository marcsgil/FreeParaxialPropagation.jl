function free_propagation_step!(ψs,phases,plan,iplan)
    plan*ψs
    map!(*,ψs,ψs,phases)
    iplan*ψs
end

function free_propagation(ψ₀,xs,ys,z::Number,k,plan,iplan)

    factor = -z/(2k)
    phases = map(ks -> cis(sum(x->factor*x^2,ks)), Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys))) |> ifftshift

    if typeof(ψ₀) <: CuArray
        phases = CuArray(phases)
    end

    cache = ifftshift(ψ₀)

    fftshift(free_propagation_step!(cache,phases,plan,iplan))
end

function free_propagation(ψ₀,xs,ys,z;k=1)
    plan = plan_fft!(ψ₀)
    iplan = plan_ifft!(ψ₀)
    free_propagation(ψ₀,xs,ys,z,k,plan,iplan)
end

function free_propagation(ψ₀,xs,ys,zs::AbstractArray;k=1)
    plan = plan_fft!(ψ₀)
    iplan = plan_ifft!(ψ₀)

    result = similar(ψ₀,size(ψ₀)...,length(zs))

    ks = Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys)) |> collect |> ifftshift

    if typeof(ψ₀) <: CuArray
        ks = CuArray(ks)
    end

    for (n,z) in enumerate(zs)
        cache = ifftshift(ψ₀)
        factor = -z/(2k)
        phases = map(ks -> cis(sum(x->factor*x^2,ks)), ks)
        free_propagation_step!(cache,phases,plan,iplan)
        fftshift!(view(result,:,:,n),cache)
    end

    result
end