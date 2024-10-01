# Common or similar methods for grid axes, grids, planar fields, etc.

# Unary plus.
Base.:(+)(A::GridAxis) = A
Base.:(+)(A::Grid) = A

# Unary minus does not change the indices, only the sign of the step.
Base.:(-)(A::GridAxis) = GridAxis(-step(A), eachindex(A))
Base.:(-)(A::Grid) = Grid(-step(A), axes(A))

# Coordinate type is a trait.
TwoDimensional.coord_type(A::Union{GridAxis,Grid,PlanarField}) = coord_type(typeof(A))
TwoDimensional.coord_type(::Type{<:GridAxis{T}}) where {T} = T
TwoDimensional.coord_type(::Type{<:Grid{T}}) where {T} = T
TwoDimensional.coord_type(::Type{<:PlanarField{T,S}}) where {T,S} = S

# Extract coordinate axes.
abscissae(A::Union{Grid,PlanarField}) = GridAxis(step(A), axes(A)[1])
ordinates(A::Union{Grid,PlanarField}) = GridAxis(step(A), axes(A)[2])

TwoDimensional.BoundingBox(A::Union{Grid,PlanarField}) = BoundingBox{coord_type(A)}(A)
TwoDimensional.BoundingBox{T}(A::Union{Grid,PlanarField}) where {T} =
    BoundingBox{T}(extrema(A.X), extrema(A.Y)) # FIXME: margin?

# Extend `view` to yield, if possible, an object of the same kind (i.e. with a given
# step). For multi-dimensional grids (grids and planar fields), only unit step sub-ranges
# or colons can yield another grid that is guaranteed to have equally spaced nodes. For
# uni-dimensional grids (grid axes), it is possible with non-unit sub-index to have a
# result which is not representable as a grid axis (due to the constraint that the grid
# coordinate is the index times the step, it is not sufficient to multiply the original
# step by the step of the sub-index range). In other words, only unit-ranges and colons
# are considered below.
Base.view(A::GridAxis, I::Colon) = A
Base.view(A::GridAxis, I::AbstractUnitRange{<:Integer}) =
    GridAxis(view(eachindex(A), I); step = step(A))

const UnitStepSubindex = Union{Colon,AbstractUnitRange{<:Integer}}

Base.view(A::Grid, I::Vararg{UnitStepSubindex,2}) =
    Grid(view(axes(A)[1], I[1]), view(axes(A)[2], I[2]); step = step(A))

Base.view(A::PlanarField, I::Vararg{UnitStepSubindex,2}) =
    PlanarField(view(parent(A), I...); step = step(A))

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
