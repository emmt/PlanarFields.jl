
const ArrayAxisLike = AbstractUnitRange{<:Integer}
const ArrayAxesLike{N} = NTuple{N,ArrayAxisLike}

"""
    PlanarFields.Angle

is the union of quantities suitable to specifiy an angle.

"""
const Angle = Unitful.Quantity{<:Real,Unitful.NoDims,<:Union{typeof(°),typeof(rad)}}

"""
    PlanarFields.ReciprocalLength

is the union of quantities suitable to specifiy a reciprocal length.

"""
const ReciprocalLength = Union{
    Unitful.Quantity{T, Unitful.𝐋^-1, U},
    Unitful.Level{L, S, Unitful.Quantity{T, Unitful.𝐋^-1, U}} where {L, S}} where {T, U}

"""
    PlanarFields.StdLength{F}

is the type of a length expressed in standard units (meters) and with
floating-point type `F`.

"""
const StdLength{F<:AbstractFloat} = Unitful.Quantity{
    F, Unitful.𝐋,
    Unitful.FreeUnits{(Unitful.Unit{:Meter,Unitful.𝐋}(0, 1//1),), Unitful.𝐋, nothing}}

"""
    PlanarFields.StdReciprocalLength{F}

is the type of a reciprocal length expressed in standard units (1/meters) and
with floating-point type `F`.

"""
const StdReciprocalLength{F<:AbstractFloat} = Unitful.Quantity{
    F, Unitful.𝐋^-1,
    Unitful.FreeUnits{(Unitful.Unit{:Meter,Unitful.𝐋}(0, -1//1),), Unitful.𝐋^-1, nothing}}

"""
    PlanarFields.StdAngle{F}

is the type of an angle expressed in standard units (radians) and with
floating-point type `F`.

"""
const StdAngle{F<:AbstractFloat} = Unitful.Quantity{F, Unitful.NoDims, typeof(rad)}

"""
    PlanarFields.Dimensionless{Real}
    PlanarFields.Dimensionless{Complex}

are unions of types suitable for dimensionless real and complex values. These
aliases are introduced to cope with dimensionless quantities that may result
from expressions involving `Unitful` quantities and that have not yet been
converted to a bare numerical type.

"""
const Dimensionless{T} = Union{T,Unitful.Quantity{<:T,Unitful.NoDims}}

"""
    GridAxis{T}

is the type of abstract ranges used to represent coordinates along Cartesian axes of grids
of equally spaced nodes. Type parameter `T` is the element-type of the range.

Compared to other base abstract range types, the indices of grid axes are not necessarily
1-based and their value is always given by multipying the index by the step: `A[i] ===
step(A)*i` holds for any valid index `i` of a grid axis `A`. This property is fundamental
to simplify checking whether fields sampled on grids of equally spaced nodes are
compatible (the steps and index ranges of the combined fields must match) and applying
discrete transforms on such fields.

"""
struct GridAxis{T<:Number,I<:ArrayAxis} <: AbstractRange{T}
    step::T
    inds::I
    function GridAxis(step::Number, inds::I) where {I<:ArrayAxis}
        T = promote_step_type(step)
        return new{T,I}(step, inds)
    end
    function GridAxis{T}(step::Number, inds::I) where {T<:Number,I<:ArrayAxis}
        check_step_type(T)
        return new{T,I}(step, inds)
    end
end

struct Grid{T<:Number,I<:ArrayAxes{2}} <: AbstractMatrix{Point{T}}
    step::T
    inds::I
    function Grid(step::Number, inds::I) where {I<:ArrayAxes{2}}
        T = promote_step_type(step)
        return new{T,I}(step, inds)
    end
    function Grid{T}(step::Number, inds::I) where {T<:Number,I<:ArrayAxes{2}}
        check_step_type(T)
        return new{T,I}(step, inds)
    end
end

struct PlanarField{T,S<:Number,L,A<:AbstractArray{T,2}} <: AbstractArray{T,2}
    vals::A
    step::S
    function PlanarField(vals::A, step::S) where {T,S<:Number,A<:AbstractArray{T,2}}
        L = IndexStyle(A) === IndexLinear()
        return new{T,S,L,A}(vals, step)
    end
end

# Union of types having a given step.
const WithStep = Union{GridAxis,Grid,PlanarField}

"""
    PlanarFields.Boxable

is the union of types of geometric objects that have a bounding-box and that can be used
to define a grid of equally spaced nodes.

"""
const Boxable = Union{ShapeElement,Mask,BoundingBox}
