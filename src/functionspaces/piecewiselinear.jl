struct PiecewiseLinear{T, S<:AbstractVector{<:Real}} <: AbstractPiecewise{TaylorSeries{T}}
    nodes::S
    values::Vector{T}
    function PiecewiseLinear(nodes::S, values::Vector{T}) where {S<:AbstractVector,T}
        @assert length(nodes) == length(values)
        @assert eltype(nodes) <: Real
        if !(nodes isa AbstractRange)
            @assert issorted(nodes)
        end
        return new{T, S}(nodes, values)
    end
end

nodes(p::PiecewiseLinear) = p.nodes
elements(p::PiecewiseLinear) = Base.Generator(i->p[i], Base.OneTo(length(p)))
nodevalues(p::PiecewiseLinear) = p.values

Base.length(p::PiecewiseLinear) = length(p.values) - 1
function Base.getindex(p::PiecewiseLinear, i)
    1 <= i <= length(p) || throw(BoundsError(p, i))
    v0 = p.values[i]
    v1 = p.values[i+1]
    x0 = nodes(p)[i]
    x1 = nodes(p)[i+1]
    dx = x1 - x0
    xmid = (x0 + x1)/2
    return TaylorSeries([(v0+v1)/2, (v1-v0)/dx], xmid)
end

function (P::PiecewiseLinear)(x)
    nodes = P.nodes
    values = P.values
    x₀ = first(nodes)
    x₁ = last(nodes)
    (x < x₀ || x > x₁) && throw(DomainError())
    x == x₀ && return first(P.values)
    x == x₁ && return last(P.values)
    if nodes isa AbstractRange
        ir = (x-x₀)/step(nodes)
        i = floor(Int, ir)
        r = ir - i
        return (one(r) - r) * values[i+1] + r * values[i+2]
    else
        j = findlast(<=(x), nodes)
        @assert j !== nothing && j !== length(nodes)
        dx = nodes[j+1] - nodes[j]
        return ((nodes[j+1] - x)/dx) * values[j] + ((x - nodes[j])/dx) * values[j+1]
    end
end

for f in (:copy, :zero, :one, :conj, :transpose, :adjoint, :real, :imag, :-)
    @eval Base.$f(p::PiecewiseLinear) = PiecewiseLinear(nodes(p), map($f, nodevalues(p)))
end

function Base.similar(p::PiecewiseLinear{T}) where T
    if isbitstype(T)
        return PiecewiseLinear(nodes(p), similar(nodevalues(p)))
    else
        return PiecewiseLinear(nodes(p), map(similar, nodevalues(p)))
    end
end

function Base.:+(p1::PiecewiseLinear, p2::PiecewiseLinear)
    @assert nodes(p1) == nodes(p2)
    return PiecewiseLinear(nodes(p1), nodevalues(p1) .+ nodevalues(p2))
end

function Base.:-(p1::PiecewiseLinear, p2::PiecewiseLinear)
    @assert nodes(p1) == nodes(p2)
    return PiecewiseLinear(nodes(p1), nodevalues(p1) .- nodevalues(p2))
end

function LinearAlgebra.rmul!(p::PiecewiseLinear, α)
    rmul!(p.values, α)
    return p
end

function LinearAlgebra.lmul!(α, p::PiecewiseLinear)
    lmul!(α, p.values)
    return p
end

function LinearAlgebra.mul!(pdst::PiecewiseLinear, α, psrc::PiecewiseLinear)
    @assert nodes(pdst) == nodes(psrc)
    mul!(pdst.values, α, psrc.values)
    return pdst
end

function LinearAlgebra.mul!(pdst::PiecewiseLinear, psrc::PiecewiseLinear, α)
    @assert nodes(pdst) == nodes(psrc)
    mul!(pdst.values, psrc.values, α)
    return pdst
end

function LinearAlgebra.axpy!(α, px::PiecewiseLinear, py::PiecewiseLinear)
    @assert nodes(px) == nodes(py)
    axpy!(α, px.values, py.values)
    return py
end
function LinearAlgebra.axpby!(α, px::PiecewiseLinear, β, py::PiecewiseLinear)
    @assert nodes(px) == nodes(py)
    axpy!(α, px.values, β, py.values)
    return py
end

const PiecewiseLinearArray = PiecewiseLinear{<:AbstractArray}
function LinearAlgebra.rmul!(p::PiecewiseLinearArray, α::Number)
    for v in nodevalues(p)
        rmul!(v, α)
    end
    return p
end

function LinearAlgebra.lmul!(α::Number, p::PiecewiseLinearArray)
    for v in nodevalues(p)
        lmul!(α, v)
    end
    return p
end

function LinearAlgebra.mul!(pdst::PiecewiseLinearArray, α, psrc::PiecewiseLinear)
    @assert nodes(pdst) == nodes(psrc)
    for (vdst, vsrc) in zip(nodevalues(pdst), nodevalues(psrc))
        mul!(vdst, α, vsrc)
    end
    return pdst
end

function LinearAlgebra.mul!(pdst::PiecewiseLinearArray, psrc::PiecewiseLinear, α)
    @assert nodes(pdst) == nodes(psrc)
    for (vdst, vsrc) in zip(nodevalues(pdst), nodevalues(psrc))
        mul!(vdst, α, vsrc)
    end
    return pdst
end

function LinearAlgebra.axpy!(α, px::PiecewiseLinear, py::PiecewiseLinearArray)
    @assert nodes(px) == nodes(py)
    for (vx, vy) in zip(nodevalues(px), nodevalues(py))
        axpy!(α, vx, vy)
    end
    return py
end
function LinearAlgebra.axpby!(α, px::PiecewiseLinear, β, py::PiecewiseLinearArray)
    @assert nodes(px) == nodes(py)
    for (vx, vy) in zip(nodevalues(px), nodevalues(py))
        axpby!(α, vx, β, vy)
    end
    return py
end
