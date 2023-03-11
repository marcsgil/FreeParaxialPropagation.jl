normalize(ψ::AbstractArray{T,2}) where T  = ψ/maximum(abs.(ψ))

function normalize(ψ::AbstractArray{T,3};normalize_by_first=false) where T
    result = similar(ψ)
    if normalize_by_first
        result = ψ/maximum(abs.(view(ψ,:,:,1)))
    else
        for n in axes(ψ,3)
            result[:,:,n] = view(ψ,:,:,n)/maximum(abs.(view(ψ,:,:,n)))
        end
    end
    result
end

function normalize(ψ::AbstractArray{T,3};normalize_by_first=false) where T <: Real
    result = similar(ψ)
    if normalize_by_first
        result = ψ/maximum(view(ψ,:,:,1))
    else
        for n in axes(ψ,3)
            result[:,:,n] = view(ψ,:,:,n)/maximum(view(ψ,:,:,n))
        end
    end
    result
end

function convert2image(ψ::AbstractArray{T,N};colormap=:hot,ratio=1) where {T,N}
    imresize(map( pixel -> get(colorschemes[colormap], pixel), abs2.(ψ) ),ratio=ratio)
end

function convert2image(ψ::AbstractArray{T,N};colormap=:hot,ratio=1) where {T <: Real ,N }
    imresize(map( pixel -> get(colorschemes[colormap], pixel), ψ ),ratio=ratio)
end

function vizualize(ψ::AbstractArray{T,2}; colormap=:hot,ratio=1) where T
    convert2image(normalize(ψ),colormap=colormap,ratio=ratio)
end

function vizualize(ψ::AbstractArray{T,3}; colormap=:hot,ratio=1,normalize_by_first=false) where T
    convert2image(hcat(eachslice(normalize(ψ,normalize_by_first=normalize_by_first),dims=3)...),colormap=colormap,ratio=ratio)
end

function vizualize(ψ::AbstractArray{T,4}; colormap=:hot,ratio=1,normalize_by_first=false) where T
    vcat(vizualize.( eachslice(ψ,dims=4),colormap=colormap,ratio=ratio,normalize_by_first=normalize_by_first )...)
end

function make_animation(ψs::AbstractArray{T,3}; colormap=:hot,ratio=1,fps=16,normalize_by_first=normalize_by_first) where T 
    ImageShow.gif([convert2image(normalize(ψ,normalize_by_first=normalize_by_first),colormap=colormap,ratio=ratio) for ψ in eachslice(ψs,dims=3)],fps=fps)
end