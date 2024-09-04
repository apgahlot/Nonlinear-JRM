# joTimeLapse operator

## helper module
module joTimeLapse_etc

    function apply_fwd(L::Integer, n::Integer, z::Vector{T}) where T
        return view(z, n+1:L*n) - repeat(view(z, 1:n), L-1)
    end 

    function apply_adj(L::Integer, n::Integer, x::Vector{T}) where T
        return vcat(-dropdims(sum(reshape(view(x,:), n, L-1), dims=2),dims=2), x)
    end

end
using .joTimeLapse_etc

export joTimeLapse

function joTimeLapse(L::Integer, nn::Integer; DDT=Float32, RDT=Float32, name::String="joTimeLapse")

    (L >= 2) || throw(joLinearFunctionException("need 2 or more surveys to take time-lapse"))
    TL = joLinearFunctionFwd((L-1)*nn, L*nn,
        v1 -> joTimeLapse_etc.apply_fwd(L, nn, v1),
        v2 -> joTimeLapse_etc.apply_adj(L, nn, v2),
        v3 -> joTimeLapse_etc.apply_adj(L, nn, v3),
        v4 -> joTimeLapse_etc.apply_fwd(L, nn, v4),
        DDT, RDT; name=name)
    return TL
end