
"""
    relative_precision(x::Number) -> r
    relative_precision(T::Type{<:Number}) -> r

yield the relative precision of the number `x`. Argument may also be a type. The result is
a real number which only depend on the bare numeric type of `x` or `T`. If the bare
numeric type is a floating-point type `F`, the result is `eps(F)`. If the bare numeric
type is an integer or rational type `R`, the result is `zero(R)`. Otherwise,
`eps(Float64)` is returned.

"""
relative_precision(x::Number) = relative_precision(typeof(x))
relative_precision(::Type{T}) where {T<:Number} = relative_precision(real_type(T))
relative_precision(::Type{T}) where {T<:Integer} = zero(T)
relative_precision(::Type{T}) where {T<:Rational} = zero(T)
relative_precision(::Type{T}) where {T<:AbstractFloat} = eps(T)
relative_precision(::Type{T}) where {T<:Real} = eps(Float64)
relative_precision(::Type{AbstractFloat}) = eps(Float64)

"""
    PlanarFields.nearly_equal(x, y) -> bool

yields whether numbers `x` and `y` are nearly equal relatively to the worst precision of
the two.

"""
function nearly_equal(x::T, y::T) where {T<:Number}
    if x == y
        return true
    else
        R = bare_type(T)
        if R <: Union{Integer,Rational}
            return false
        else
            prec = relative_precision(R)
            return abs(x - y) ≤ prec*max(abs(x), abs(y))
        end
    end
end

function nearly_equal(x::Tx, y::Ty) where {Tx<:Number,Ty<:Number}
    if x == y
        return true
    else
        Rx, Ry = bare_type(Tx), bare_type(Ty)
        if (Rx <: Union{Integer,Rational}) & (Ry <: Union{Integer,Rational})
            return false
        else
            # use the worst relative precision of the two
            prec = max(relative_precision(Rx), relative_precision(Ry))
            return abs(x - y) ≤ prec*max(abs(x), abs(y))
        end
    end
end
