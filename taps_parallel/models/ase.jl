
using PyCall
export ASE

""" ASE interface simple


example
=======

>>> asemodel = ASE(("Al12Au",), Dict(:cell=>cell), "ase.calculators.emt", "EMT", (), Dict())

ASE(PyObject Atoms(symbols='Al12Au', pbc=False, cell=[5.727564927611035, 5.727564927611035, 13.75],
calculator=EMT(...)), PyObject <ase.calculators.emt.EMT object at 0x2b55d12a94d0>)
"""
struct ASE <: Model
    atoms
    calc
    results
end

function ASE(atoms_args, atoms_kwargs, calc_module, calc_name, calc_args, calc_kwargs)

    atoms = pyimport("ase.atoms")[:Atoms](atoms_args...; atoms_kwargs...)
    calc = pyimport(calc_module)[Symbol(calc_name)](calc_args...; calc_kwargs...)
    atoms.set_calculator(calc)
    ASE(atoms, calc, Dict())
end


@inline (model::ASE)(paths::Paths, coords::Coords, properties::Array{T, 1}) where {T} = (model::ASE)(coords::Coords, Array{String, 1}(properties))
@inline (model::ASE)(coords::Coords, properties::Array{T, 1}) where {T} = (model::ASE)(convert(Cartesian{eltype(coords), 3}, coords), Array{String, 1}(properties))

"""
coords : Cartesian; N x 3

return : N x size(property) results
"""
function (model::ASE)(coords::Cartesian{T, 3}, properties::Array{String, 1}) where {T<:Number}
    N, A, _ = size(coords)
    results = Dict()
    cache = Dict([(property, []) for property in properties])

    atoms = model.atoms

    # Initialize

    for i in 1:N
        positions = coords.coords[i, :, :]
        atoms.set_positions(positions)

        if "potential" in properties
            push!(cache["potential"], atoms.get_potential_energy())
        end

        if "gradients" in properties
            dV = -atoms.get_forces()
            push!(cache["gradients"], reshape(dV, 1, size(dV)...))
        end
    end

    for property in properties
        shape = size(cache[property][1])
        if shape == ()
            results[property] = cache[property]
        else
            results[property] = cat(cache[property]..., dims=1)
        end
    end
    merge!(model.results, results)

    return results
end
