# joJIM operator

## helper module
module joJIM_etc

    function apply_fwd(L::Integer, z::Vector{T}) where T
        return @views vec(cumsum(reshape(z, :, L); dims=2))    # get original components
    end

    function apply_adj(L::Integer, x::Vector{T}) where T
        return @views vec(reverse(cumsum(reverse(reshape(x, :, L); dims=2); dims=2), dims=2))
    end

end
using .joJIM_etc

export joJIM
"""
    julia> op = joJIM(ops;[name=...])

JIM operator of an array of JOLI operators

# Signature

    joJIM(ops::Vector{joAbstractLinearOperator}; γ::Number=1f0, name::String="joJRM")

# Arguments

- `ops`: an array of JOLI operators (subtypes of joAbstractLinearOperator)
- keywords:
    - `name`: custom name

# Notes

- the domain and range types of joJIM are equal to domain type of any operator in ops
- all operators in the array ops must have the same domain/range types

# Example

define operators

    ops = [joDiag(randn(Float32,3); DDT=Float32, RDT=Float32) for i = 1:4]

define JIM operator

    A=joJIM(ops)

"""


function joJIM(ops::Vector{T1}; name::String="joJIM") where {T1<:joAbstractOperator}
    return joJIM(ops...; name=name)
end

function joJIM(ops::Vararg{T1, L}; name::String="joJIM") where {T1<:joAbstractOperator, L}
    
    # check if all operators have the same domain/range types
    isempty(ops) && throw(joLinearFunctionException("empty argument list"))
    DDT = deltype(ops[1])
    RDT = reltype(ops[1])
    #(T2 == DDT) || throw(joLinearFunctionException("weight on common component type mismatch with domain/range type"))
    opn = ops[1].n
    for i=2:L
        deltype(ops[i])==DDT || throw(joLinearFunctionException("domain type mismatch"))
        reltype(ops[i])==RDT || throw(joLinearFunctionException("range type mismatch"))
        ops[i].n==opn || throw(joLinearFunctionException("domain length mismatch"))
    end
    As = joCoreBlock(ops...)
    Φ = joLinearFunctionFwd(L*ops[1].n, L*ops[1].n,
        v1 -> joJIM_etc.apply_fwd(L, v1),
        v2 -> joJIM_etc.apply_adj(L, v2),
        v3 -> joJIM_etc.apply_adj(L, v3),
        v4 -> joJIM_etc.apply_fwd(L, v4),
        DDT, DDT; name=name)
    return As * Φ
end

function joJIM(L::Integer, nn::Integer; DDT=Float32, RDT=Float32, name::String="joJIM")

    Φ = joLinearFunctionFwd(L*nn, L*nn,
        v1 -> joJIM_etc.apply_fwd(L, v1),
        v2 -> joJIM_etc.apply_adj(L, v2),
        v3 -> joJIM_etc.apply_adj(L, v3),
        v4 -> joJIM_etc.apply_fwd(L, v4),
        DDT, RDT; name=name)
    return Φ

end