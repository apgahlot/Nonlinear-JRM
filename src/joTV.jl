# joTV operator

## helper module
module joTV_etc

    function apply_fwd(n::Tuple{Integer, Integer}, z::Vector{T}; d::Tuple{T, T}=(1f0, 1f0)) where T
        return vcat(vec(diff(reshape(view(z,:),n), dims=2))./d[2],vec(diff(reshape(view(z,:),n), dims=1))./d[1])
    end 

    function apply_adj(n::Tuple{Integer, Integer}, x::Vector{T}; d::Tuple{T, T}=(1f0, 1f0)) where T
        function adjoint_diff(n1::Integer, z::AbstractMatrix{T}) where T
            return vcat(-view(z, 1:1, :), view(z, 1:n1-1, :)-view(z, 2:n1, :), view(z, n1:n1, :))
        end
        return vcat(-view(x, 1:n[1]),
        view(x, 1:(n[2]-2)*n[1])-view(x, n[1]+1:(n[2]-1)*n[1]),
        view(x, (n[2]-2)*n[1]+1:(n[2]-1)*n[1]))/d[2] + 
        vec(adjoint_diff(n[1]-1, reshape(view(x, (n[2]-1)*n[1]+1:(n[2]-1)*n[1]+(n[1]-1)*n[2]), n[1]-1, n[2])))/d[1]
    end

end
using .joTV_etc

export joTV

function joTV(n::Tuple{Integer, Integer}; d::Tuple{T, T}=(1f0, 1f0), DDT=Float32, RDT=Float32, name::String="joTV") where T

    TL = joLinearFunctionFwd((n[1]-1)*n[2]+n[1]*(n[2]-1), prod(n),
        v1 -> joTV_etc.apply_fwd(n, v1; d=DDT.(d)),
        v2 -> joTV_etc.apply_adj(n, v2; d=DDT.(d)),
        v3 -> joTV_etc.apply_adj(n, v3; d=DDT.(d)),
        v4 -> joTV_etc.apply_fwd(n, v4; d=DDT.(d)),
        DDT, RDT; name=name)
    return TL
end
