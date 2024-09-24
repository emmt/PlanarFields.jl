"""
    A = PlanarField{T}(grid)
    A = PlanarField{T,S}(grid)

Allocate an object `A` for storing the values of a field sampled on the 2-dimensional
planar `grid` of equally spaced nodes. `T` is the type of the field values, `S` is the
type of node coordinates (the same as for `grid` by default). The values stored by the
returned array are undefined.

Another posibility is to specify any arguments and/or keywords that are accepted by the
[`Grid`](@ref) constructor. For example:

    A = PlanarField{T[,S]}(X, Y)
    A = PlanarField{T[,S]}(step, I, J)
    A = PlanarField{T[,S]}(I, J, step)
    A = PlanarField{T[,S]}(I, J; step)

where `X` and `Y` are the grid abscissae and ordinates, or where `I`, `J`, and `step` are
the index ranges along the array dimensions and `step` is the grid node spacing. The type
`T` of the field values must be specified. If the type `S` of the node coordinates is not
specified, it is inferred for that of `X` and `Y` or from `step`.

A planar field object `A` is an abstract 2-dimensional array whose values can be accessed
by `A[I]`. The position of a node at Cartesian index `I` is given by `step*Point(I)`.

"""
PlanarField{T}(grid::Grid) where {T} =
    PlanarField(OffsetArray(Array{T}(undef, size(grid)), axes(grid)), step(grid))

# Build a planar grid from arguments and keywords.
PlanarField{T}(args...; kwds...) where {T} = PlanarField{T}(Grid(args...; kwds...))
PlanarField{T,S}(args...; kwds...) where {T,S<:Number} = PlanarField{T}(Grid{S}(args...; kwds...))

"""
    A = PlanarField{T=eltype(vals),S=typeof(step)}(vals, step)
    A = PlanarField{T=eltype(vals),S=typeof(step)}(vals; step)

Build an object `A` representing a field sampled on a 2-dimensional planar grid of equally
spaced nodes. `vals` is the 2-dimensional array storing the sampled field values and
`step` is the node spacing. Both `vals` and `step` may have units.

If the specified type `T` of the field values is the same as `eltype(vals)` (the default),
the returned object `A` and `vals` share their values; otherwise, the values of `A` are
stored in a different array than `vals`. The method `parent(A)` yields the array
backing the storage of the values in `A`.

"""
PlanarField(vals::AbstractMatrix; step::Number) = PlanarField(vals, step)

PlanarField{T}(vals::AbstractMatrix; step::Number) where {T} = PlanarField{T}(vals, step)
PlanarField{T}(vals::AbstractMatrix{T}, step::Number) where {T} = PlanarField(vals, step)
PlanarField{T}(vals::AbstractMatrix, step::Number) where {T} =
    PlanarField(copyto!(similar(vals, T), vals), step)

PlanarField{T,S}(vals::AbstractMatrix; step::Number) where {T,S<:Number} =
    PlanarField{T,S}(vals, step)
PlanarField{T,S}(vals::AbstractMatrix, step::Number) where {T,S<:Number} =
    PlanarField{T}(vals, as(T, step))

# Copy/convert constructors.
PlanarField(A::PlanarField) = A
PlanarField{T}(A::PlanarField{T}) where {T} = A
PlanarField{T}(A::PlanarField) where {T<:Number} = PlanarField{T}(parent(A), step(A))
PlanarField{T,S}(A::PlanarField{T,S}) where {T,S<:Number} = A
PlanarField{T,S}(A::PlanarField) where {T,S<:Number} = PlanarField{T,S}(parent(A), step(A))
Base.convert(::Type{T}, A::T) where {T<:PlanarField} = A
Base.convert(::Type{T}, A::PlanarField) where {T<:PlanarField} = T(A)
Base.copy(A::PlanarField) = PlanarField(copy(parent(A)), step(A))

# The base similar method takes an additional step keyword for a planar field.
Base.similar(A::PlanarField; kwds...) = similar(A, eltype(A), axes(A); kwds...)
Base.similar(A::PlanarField, ::Type{T}; kwds...) where {T} = similar(A, T, axes(A); kwds...)
Base.similar(A::PlanarField, inds::NTuple{2,Union{Integer,ArrayAxisLike}}; kwds...) =
    similar(A, eltype(A), inds; kwds...)
function Base.similar(A::PlanarField, ::Type{T}, inds::NTuple{2,Union{Integer,ArrayAxisLike}};
                      step::Number = step(A)) where {T}
    return PlanarField{T}(Grid(step, to_axes(inds)))
end

# Retrieve the grid of the nodes.
Grid(A::PlanarField) = Grid(step(A), axes(A))
Grid{S}(A::PlanarField) where {S<:Number} = Grid{S}(step(A), axes(A))

# Accessors.
Base.parent(A::PlanarField) = getfield(A, :vals)
Base.step(A::PlanarField) = getfield(A, :step)

# Traits.
TwoDimensional.coord_type(A::PlanarField) = coord_type(typeof(A))
TwoDimensional.coord_type(::Type{<:PlanarField{T,S}}) where {T,S} = S

# Abstract array API for PlanarField objects.
Base.length(A::PlanarField) = prod(size(A))
Base.size(A::PlanarField) = map(length, axes(A))
Base.axes(A::PlanarField) = axes(parent(A))
Base.IndexStyle(::Type{PlanarField{T,S,true,A}}) where {T,S,A} = IndexLinear()
Base.IndexStyle(::Type{PlanarField{T,S,false,A}}) where {T,S,A} = IndexCartesian()

@inline function Base.getindex(A::PlanarField{T,S,true}, i::Int) where {T,S}
    @boundscheck checkbounds(A, i)
    return @inbounds getindex(parent(A), i)
end

@inline function Base.getindex(A::PlanarField{T,S,false}, I::Vararg{Int,2}) where {T,S}
    @boundscheck checkbounds(A, I...)
    return @inbounds getindex(parent(A), I...)
end

@inline function Base.setindex!(A::PlanarField{T,S,true}, x, i::Int) where {T,S}
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(parent(A), x, i)
    return A
end

@inline function Base.setindex!(A::PlanarField{T,S,false}, x, I::Vararg{Int,2}) where {T,S}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(mask(A), x, I...)
    return A
end

# Implement broadcasting for planar fields (see Julia doc. "Interfaces, Customizing
# broadcasting"). This turns out to be quite simple: we define a custom broadcast style
# and extend `similar` for that broadcasting style so as to determine the common step of
# the broadcast arguments.
Base.BroadcastStyle(::Type{<:PlanarField}) = Broadcast.ArrayStyle{PlanarField}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PlanarField}},
                      ::Type{T}) where {T}
    # Scan the inputs for the grid step.
    stp = broadcast_step(bc)
    return PlanarField{T}(axes(bc); step = stp)
end

# Find suitable common step in broadcasting.
broadcast_step(bc::Base.Broadcast.Broadcasted) = broadcast_step(nothing, bc.args)
broadcast_step(s::Union{Nothing,Number}, args::Tuple) = broadcast_step(broadcast_step(s, args[1]), Base.tail(args))
broadcast_step(s::Union{Nothing,Number}, A::Number) = s
broadcast_step(s::Nothing, A::PlanarField) = step(A)
broadcast_step(s::Number, A::PlanarField) = common_step(s, step(A))
broadcast_step(s::Union{Nothing,Number}, ::Tuple{}) = s
broadcast_step(s::Union{Nothing,Number}, A::Any) = throw(ArgumentError(
    "planar fields can only be combined with planar fields and numbers, got instance of `$(typeof(A))`"))

"""
    PlanarFields.common_step(s1, s2) -> s

yields the step `s` resulting from combining two grids, grid axes, or planar fields with
steps `s1` and `s2`. The result `s` has the concrete promoted type of `s1` and `s2` and
the value of the most precise of `s1` and `s2` (the average of `s1` and `s2` is returned
if they are different but have the same precision). An exception is thrown if `s1` and `s2`
cannot be converted to a common concrete type or if `s1` and `s2` are not nearly the same
(relative to the worst precision of the two).

Arguments may also be two instances of objects having a defined grid step.

"""
function common_step(s1::T1, s2::T2) where {T1<:Number,T2<:Number}
    T = to_same_concrete_type(T1, T2) # type of the result
    r1, r2 = as(T, s1), as(T, s2)
    if r1 === r2
        return r1
    else
        R1, R2 = bare_type(T1), bare_type(T2)
        if !((R1 <: Union{Integer,Rational}) & (R1 <: Union{Integer,Rational}))
            # At least one of the steps is floating-point or irrational.
            prec1, prec2 = relative_precision(R1), relative_precision(R2)
            if prec1 < prec2
                # s1 has the strictly highest precision
                if abs(r1 - r2) ≤ prec2*max(abs(r1), abs(r2))
                    return r1
                end
            elseif abs(r1 - r2) ≤ prec1*max(abs(r1), abs(r2))
                if prec2 < prec1
                    return r2
                else
                    return as(T, (r1 + r2)/2)
                end
            end
        end
    end
    throw_not_nearly_equal_steps(s1, s2)
end

@noinline throw_not_nearly_equal_steps(s::Number...) =
    throw(ArgumentError("steps are not nearly equal"))

common_step(A::WithStep, B::WithStep) = common_step(step(A), step(B))

nearly_same_step(A::WithStep) = true
nearly_same_step(A::WithStep, B::WithStep) = nearly_same_step(step(A), step(B))
function nearly_same_step(a::Number, b::Number)
    a == b && return true
    # consider the worst relative precision of the two
    e = max(relative_precision(a), relative_precision(b))
    return abs(a - b) ≤ e*max(abs(a), abs(b))
end

"""
    PlanarField{T,S}(mask; step, kwds...)
    PlanarField{T,S}(mask, step; kwds...)
    PlanarField{T,S}(step, mask; kwds...)

Create a planar fields from a 2-dimensional `mask` by sampling the mask over a grid of
equally spaced nodes. The grid spacing, `step`, is specified as a positional argument or
as a keyword.

If `S`, the coordinate type, is not specified, `S = typeof(step)` is assumed.

If `T`, the field element type, is not specified, `S = floating_point_type(step)` is
assumed.

The bounding-box of the mask and the node spacing are used to determine the indices of the
grid as follows:

   grid = Grid{S}(mask; step=step, margin=margin)

where keyword `margin` is an optional keywords. Other keywords are passed to
`TwoDimensional.forge_mask!`.

"""
function PlanarField(mask::Union{MaskElement,Mask}; step::Number, kwds...)
    T = floating_point_type(step)
    return PlanarField{T}(mask; step = step, kwds...)
end

PlanarField(mask::Union{MaskElement,Mask}, step::Number; kwds...) =
    PlanarField(mask; step = step, kwds...)

PlanarField(step::Number, mask::Union{MaskElement,Mask}; kwds...) =
    PlanarField{T}(mask; step = step, kwds...)

function PlanarField{T}(mask::Union{MaskElement,Mask}; step::Number, kwds...) where {T}
    S = typeof(step)
    return PlanarField{T,S}(mask; step = step, kwds...)
end

PlanarField{T}(mask::Union{MaskElement,Mask}, step::Number; kwds...) where {T} =
    PlanarField{T}(mask; step = step, kwds...)

PlanarField{T}(step::Number, mask::Union{MaskElement,Mask}; kwds...) where {T} =
    PlanarField{T}(mask; step = step, kwds...)

function PlanarField{T,S}(mask::Union{MaskElement,Mask}; step::Number,
                          margin=default_margin, kwds...) where {T,S<:Number}
    box = BoundingBox(mask)
    grid = Grid{S}(box; step=step, margin=margin)
    return  PlanarField{T}(grid, mask; kwds...)
end

PlanarField{T,S}(mask::Union{MaskElement,Mask}, step::Number; kwds...) where {T,S<:Number} =
    PlanarField{T,S}(mask; step = step, kwds...)

PlanarField{T,S}(step::Number, mask::Union{MaskElement,Mask}; kwds...) where {T,S<:Number} =
    PlanarField{T,S}(mask; step = step, kwds...)

"""
    PlanarField{T,S}(grid, mask; kwds...)
    PlanarField{T,S}(mask, grid; kwds...)

Create a planar fields from a 2-dimensional `mask` by sampling the mask over the given `grid` of
equally spaced nodes.

If `S`, the coordinate type, is not specified, `S = coord_type(grid)` is assumed.

If `T`, the field element type, is not specified, `S = floating_point_type(step)` is
assumed.

Keywords `kwds...` are passed to `TwoDimensional.forge_mask!`.

"""
PlanarField{T,S}(grid::Grid, mask::Union{MaskElement,Mask}; kwds...) where {T,S<:Number} =
    PlanarField{T}(Grid{S}(grid), mask; kwds...)

PlanarField{T,S}(mask::Union{MaskElement,Mask}, grid::Grid; kwds...) where {T,S<:Number} =
    PlanarField{T,S}(grid, mask; kwds...)

function PlanarField{T}(grid::Grid, mask::Union{MaskElement,Mask}; kwds...) where {T}
    A = PlanarField{T}(grid)
    return forge_mask!(A, grid.X, grid.Y, mask; kwds...)
end

PlanarField{T}(mask::Union{MaskElement,Mask}, grid::Grid; kwds...) where {T} =
    PlanarField{T}(grid, mask; kwds...)
