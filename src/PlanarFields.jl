module PlanarFields

export
    Grid,
    GridAxis,
    PlanarField

using TypeUtils
using OffsetArrays

using Unitful
using Unitful:
    Length,
    °, rad,
    km, m, cm, mm, µm, nm

using TwoDimensional
using TwoDimensional: Mask, MaskElement, ShapeElement, dot, ⋅

using Base: axes1
using Base.Broadcast: broadcasted

const default_margin = 1//2

include("types.jl")
include("utils.jl")
include("grids.jl")
include("fields.jl")
include("common.jl")

end
