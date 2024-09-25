"""
    A = GridAxis{T}(step, inds)
    A = GridAxis{T}(inds, step)
    A = GridAxis{T}(inds; step)

Build a grid axis object with element type `T`, indices `inds` and spacing `step`. If `T`
is not specified, it is inferred from the type of `step` promoted to be stable by the
multiplication by an `Int`.

A grid axis object `A` is an abstract range such that `A[i]` yields `step*i`.

Examples:

    using Unitful
    A = GridAxis(0.1u"mm", -5:11)

"""
GridAxis(rng::ArrayAxisLike; step::Number) = GridAxis(step, rng)
GridAxis(rng::ArrayAxisLike, step::Number) = GridAxis(step, rng)
GridAxis(step::Number, rng::ArrayAxisLike) = GridAxis(step, as_array_axis(rng))

GridAxis{T}(rng::ArrayAxisLike; step::Number) where {T<:Number} = GridAxis{T}(step, rng)
GridAxis{T}(rng::ArrayAxisLike, step::Number) where {T<:Number} = GridAxis{T}(step, rng)
GridAxis{T}(step::Number, rng::ArrayAxisLike) where {T<:Number} = GridAxis{T}(step, as_array_axis(rng))

# Copy/convert constructors for GridAxis objects.
GridAxis(A::GridAxis) = A
GridAxis{T}(A::GridAxis{T}) where {T<:Number} = A
GridAxis{T}(A::GridAxis) where {T<:Number} = GridAxis{T}(step(A), eachindex(A))
Base.convert(::Type{T}, A::T) where {T<:GridAxis} = A
Base.convert(::Type{T}, A::GridAxis) where {T<:GridAxis} = T(A)
Base.copy(A::GridAxis) = A

# Accessors for GridAxis objects. NOTE `eachindex(A)` for an abstract vector `A` calls
# `Base.axes1(A)`.
@inline Base.step(A::GridAxis) = getfield(A, :step)
@inline Base.axes1(A::GridAxis) = getfield(A, :inds)

# Abstract array API for GridAxis objects.
Base.length(A::GridAxis) = length(eachindex(A))
Base.size(A::GridAxis) = (length(A),)
Base.axes(A::GridAxis) = (eachindex(A),)
Base.IndexStyle(::Type{<:GridAxis}) = IndexLinear()
@inline function Base.getindex(A::GridAxis{T}, i::Int) where {T}
    @boundscheck checkbounds(A, i)
    return as(T, i*step(A))
end

# Extend methods from other packages.
TwoDimensional.coord_type(A::GridAxis) = coord_type(typeof(A))
TwoDimensional.coord_type(::Type{<:GridAxis{T}}) where {T} = T

# Unary plus.
Base.:(+)(A::GridAxis) = A
Base.:(+)(A::Grid) = A

# Unary minus does not change the indices, only the sign of the step.
Base.:(-)(A::GridAxis) = GridAxis(-step(A), eachindex(A))
Base.:(-)(A::Grid) = Grid(-step(A), axes(A))

# Extend element type conversion by `map` or by broadcasted call to numeric type
# constructor.
Broadcast.broadcasted(::Type{T}, A::GridAxis) where {T<:Number} = map(T, A)
Base.map(::Type{T}, A::GridAxis{T}) where {T<:Number} = A
Base.map(::Type{T}, A::GridAxis) where {T<:Number} =
    GridAxis(as(T, step(A)), eachindex(A))

# Speedup multiplication and division of a grid or grid axis by a number.
for (type, indices) in ((:Grid,     :axes),
                        (:GridAxis, :eachindex))
    @eval begin
        # Multiplication by a number.
        Base.:(*)(A::$type, x::Number) = *(x, A)
        Base.:(*)(x::Number, A::$type) = $type(x*step(A), $indices(A))
        Broadcast.broadcasted(::typeof(*), A::$type, x::Number) = *(A, x)
        Broadcast.broadcasted(::typeof(*), x::Number, A::$type) = *(x, A)

        # Division by a number.
        Base.:(/)(A::$type, x::Number) = $type(step(A)/x, $indices(A))
        Base.:(\)(x::Number, A::$type) = A/x
        Broadcast.broadcasted(::typeof(/), A::$type, x::Number) = A/x
        Broadcast.broadcasted(::typeof(\), x::Number, A::$type) = x\A
    end
end

# Broadcasted addition of a number is a bit more complex because it must be checked
# that the offset is a multiple of the step.
Broadcast.broadcasted(::typeof(+), off::Number, A::GridAxis) = broadcasted(+, A, off)
function Broadcast.broadcasted(::typeof(+), A::GridAxis, off::Number)
    off, stp = to_same_type(off, step(A))
    if bare_type(off) <: Integer
        iszero(rem(off, stp)) || error("offset is not a multiple of the step")
        i = div(off, stp)
    else
        d = off/stp
        r = round(d)
        nearly_equal(r, d) || error("offset is not nearly a multiple of the step")
        i = Int(r) # FIXME this may clash if `r` is too large for an `Int`
    end
    return GridAxis(stp, eachindex(A) .+ i)
end

# Broadcasted subtraction of a number.
Broadcast.broadcasted(::typeof(-), off::Number, A::GridAxis) = broadcasted(+, off, -A)
Broadcast.broadcasted(::typeof(-), A::GridAxis, off::Number) = broadcasted(+, A, -off)

# Broadcasted addition of a point to a grid.
Broadcast.broadcasted(::typeof(+), off::Point, A::Grid) = broadcasted(+, A, off)
Broadcast.broadcasted(::typeof(+), A::Grid, off::Point) =
    Grid(broadcasted(+, A.X, off.x), broadcasted(+, A.Y, off.y))
Base.:(+)(off::Point, A::Grid) = broadcasted(+, A, off)
Base.:(+)(A::Grid, off::Point) = Grid(A.X .+ off.x, A.Y .+ off.y)

step_tolerance(step::Number) = eps(floating_point_type(step))*abs(step)

# Extend element type conversion by `map` or by broadcasted call to Point type
# constructor.
Broadcast.broadcasted(::Type{Point{T}}, A::Grid) where {T<:Number} = map(Point{T}, A)
Base.map(::Type{Point{T}}, A::Grid{T}) where {T<:Number} = A
Base.map(::Type{Point{T}}, A::Grid) where {T<:Number} = Grid(as(T, step(A)), axes(A))

"""
    G = Grid{T}(step, (I, J))
    G = Grid{T}((I, J), step)
    G = Grid{T}(I, J; step)
    G = Grid{T}((I, J); step)
    G = Grid{T}(X, Y)
    G = Grid{T}((X, Y))

Build a 2-dimensional grid of equally spaced nodes. Possible arguments are:

* `I` and `J` are index ranges (with unit step) defining the axes of the grid and `step`
  (specified as a keyword or as a positional argument) is the spacing of the nodes along
  each dimension.

* `X` and `Y` are vectors of equally spaced coordinates along the 1st and 2nd dimension of
  the grid.

Type parameter `T` is to specify the type (possibly with units) of the coordinates. If `T`
is not specified, it is inferred from the grid step.

A grid object is an abstract matrix with Cartesian indexing and whose element type is
`Point{T}`.

The following properties are available:

    G.step    # the grid node spacing
    G.X       # the abscissae of the nodes
    G.Y       # the ordinates of the nodes

the abscissae and ordinates of the nodes are returned as abstract vectors which are
instances of [`GridAxis`](@ref).

"""
Grid(I::ArrayAxisLike, J::ArrayAxisLike; step::Number) = Grid(step, (I, J))
Grid(I::ArrayAxisLike, J::ArrayAxisLike, step::Number) = Grid(step, (I, J))
Grid(step::Number, I::ArrayAxisLike, J::ArrayAxisLike) = Grid(step, (I, J))
Grid{T}(I::ArrayAxisLike, J::ArrayAxisLike; step::Number) where {T<:Number} = Grid{T}(step, (I, J))
Grid{T}(I::ArrayAxisLike, J::ArrayAxisLike, step::Number) where {T<:Number} = Grid{T}(step, (I, J))
Grid{T}(step::Number, I::ArrayAxisLike, J::ArrayAxisLike) where {T<:Number} = Grid{T}(step, (I, J))

Grid(inds::ArrayAxesLike{2}; step::Number) = Grid(step, inds)
Grid(inds::ArrayAxesLike{2}, step::Number) = Grid(step, inds)
Grid(step::Number, inds::ArrayAxesLike{2}) = Grid(step, as_array_axes(inds))
Grid{T}(inds::ArrayAxesLike{2}; step::Number) where {T<:Number} = Grid{T}(step, inds)
Grid{T}(inds::ArrayAxesLike{2}, step::Number) where {T<:Number} = Grid{T}(step, inds)
Grid{T}(step::Number, inds::ArrayAxesLike{2}) where {T<:Number} = Grid{T}(step, as_array_axes(inds))

Grid((X, Y)::NTuple{2,GridAxis}) = Grid(X, Y)
function Grid(X::GridAxis, Y::GridAxis)
    Xstep, Ystep = promote(step(X), step(Y))
    Xstep === Ystep || throw(ArgumentError("`X` and `Y` must have the same step"))
    return Grid(Xstep, (eachindex(X), eachindex(Y)))
end

Grid{T}((X, Y)::NTuple{2,GridAxis}) where {T<:Number} = Grid{T}(X, Y)
function Grid{T}(X::GridAxis, Y::GridAxis) where {T<:Number}
    Xstep, Ystep = as(T, step(X)), as(T, step(Y))
    Xstep === Ystep || throw(ArgumentError("`X` and `Y` must have the same step"))
    return Grid(Xstep, (eachindex(X), eachindex(Y)))
end

# Copy/convert constructors.
Grid(G::Grid) = G
Grid{T}(G::Grid{T}) where {T<:Number} = G
Grid{T}(G::Grid) where {T<:Number} = Grid{T}(step(G), axes(G))
Base.convert(::Type{T}, G::T) where {T<:Grid} = G
Base.convert(::Type{T}, G::Grid) where {T<:Grid} = T(G)
Base.copy(G::Grid) = G

# Accessors for Grid objects.
Base.step(G::Grid) = getfield(G, :step)
abscissae(G::Grid) = GridAxis(step(G), axes(G)[1])
ordinates(G::Grid) = GridAxis(step(G), axes(G)[2])

# Properties of Grid objects.
Base.propertynames(::Grid) = (:step, :X, :Y)
Base.getproperty(G::Grid, key::Symbol) =
    key === :step ? step(G) :
    key === :X    ? abscissae(G) :
    key === :Y    ? ordinates(G) :
    throw(KeyError(key))

# Abstract array API for Grid objects.
Base.length(A::Grid) = prod(size(A))
Base.size(A::Grid) = map(length, axes(A))
Base.axes(A::Grid) = getfield(A, :inds)
Base.IndexStyle(::Type{<:Grid}) = IndexCartesian()
@inline function Base.getindex(A::Grid{T}, I::Vararg{Int,2}) where {T}
    @boundscheck checkbounds(A, I...)
    return Point{T}(step(A)*I[1], step(A)*I[2])
end

# Extend methods from other packages.
TwoDimensional.coord_type(G::Grid) = coord_type(typeof(G))
TwoDimensional.coord_type(::Type{<:Grid{T}}) where {T} = T

"""
    G = Grid{T}(step, arr)
    G = Grid{T}(arr, step)
    G = Grid{T}(arr; step)

Build a 2-dimensional grid of nodes equally spaced by `step` and with the same indices as
array `arr`. Optional type parameter `T` is the coordinate type of the grid nodes. By
default, `T = typeof(step)`.

"""
Grid{T}(step::Number, arr::AbstractMatrix) where {T<:Number} = Grid(as(T, step), axes(arr))
Grid{T}(arr::AbstractMatrix, step::Number) where {T<:Number} = Grid{T}(step, arr)
Grid{T}(arr::AbstractMatrix; step::Number) where {T<:Number} = Grid{T}(step, arr)

Grid(step::Number, arr::AbstractMatrix) = Grid(step, axes(arr))
Grid(arr::AbstractMatrix; step::Number) = Grid(step, arr)
Grid(arr::AbstractMatrix, step::Number) = Grid(step, arr)

"""
    G = Grid{T}(step, obj; margin=$(default_margin))
    G = Grid{T}(obj, step; margin=$(default_margin))
    G = Grid{T}(obj; step, margin=$(default_margin))

Build a 2-dimensional grid of equally spaced nodes for sampling the geometric object `obj`
(e.g. a shaped object, a mask element, a mask, a bounding box, etc.) with a given `step`.
The grid is large enough to encompass the object with some margin(s) specified as a number
of grid nodes by the `margin` keyword. The default margin of `$(default_margin)` is to
ensure that the object is not cropped when it is sampled on the grid with antialiasing.
The object may be cropped if any of the margin is less than `1//2`.

Optional type parameter `T` is the coordinate type of the grid nodes. By default, `T` is
inferred from the types of the step and of the coordinates of `obj`.

"""
function Grid(step::Number, box::BoundingBox; kwds...)
    T = promote_type(coord_type(box), typeof(step))
    return Grid{T}(step, box; kwds...)
end

function Grid{T}(step::Number, box::BoundingBox; margin::Real = default_margin) where {T<:Number}
    step = as(T, step)
    I = floor(Int, as(T, box.xmin)/step - margin) : ceil(Int, as(T, box.xmax)/step + margin)
    J = floor(Int, as(T, box.ymin)/step - margin) : ceil(Int, as(T, box.ymax)/step + margin)
    return Grid(step, I, J)
end

Grid(obj::Boxable, step::Number, kwds...) = Grid(step, obj; kwds...)
Grid(obj::Boxable; step::Number, kwds...) = Grid(step, obj; kwds...)

Grid{T}(obj::Boxable, step::Number, kwds...) where {T<:Number} = Grid{T}(step, obj; kwds...)
Grid{T}(obj::Boxable; step::Number, kwds...) where {T<:Number} = Grid{T}(step, obj; kwds...)

Grid(step::Number, obj::Boxable; kwds...) = Grid(step, BoundingBox(obj); kwds...)
Grid{T}(step::Number, obj::Boxable; kwds...) where {T<:Number} =
    Grid{T}(step, BoundingBox(obj); kwds...)

"""
    T = PlanarFields.promote_step_type(step)

Promote axis or grid step type knowing that axis and grid coordinates are generated by
multiplying an `Int` index by the step. Argument may be the step value or its type. This
helper function is needed by the inner constructors of [`GridAxis`](@ref) and of
[`Grid`](@ref).

"""
@inline promote_step_type(step::Number) = promote_step_type(typeof(step))
@generated function promote_step_type(::Type{T}) where {T<:Number}
    S = typeof(zero(Int)*zero(T))
    -oneunit(S) < zero(S) || throw(ArgumentError("grid step type `T = $T` is unsigned"))
    return S
end

"""
    PlanarFields.check_step_type(step)

Check axis or grid step type knowing that axis and grid coordinates are generated by
multiplying an `Int` index by the step. Argument may be the step value or its type. This
helper function is needed by the inner constructors of [`GridAxis`](@ref) and of
[`Grid`](@ref).

"""
@inline check_step_type(step::Number) = check_step_type(typeof(step))
@generated function check_step_type(::Type{T}) where {T<:Number}
    isconcretetype(T) || throw(ArgumentError("grid step type `T = $T` is not a concrete type"))
    T === typeof(zero(Int)*zero(T)) || throw(ArgumentError(
            "grid step type `T = $T` is not stable by the multiplication by an `Int`"))
    -oneunit(T) < zero(T) || throw(ArgumentError("grid step type `T = $T` is unsigned"))
    return nothing
end
