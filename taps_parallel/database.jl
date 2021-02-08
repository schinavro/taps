module Database

using SQLite, DataFrames

export read_data, shape_x, shape_y

function read_data(filename::String, data_ids)
    db = SQLite.DB(filename)
    db = DBInterface.connect(SQLite.DB, filename)
    img_ids = data_ids["image"]
    N = size(img_ids)
    cmdline = "SELECT * FROM image WHERE rowid IN ("
    cmdline *= join(map(id -> string(id), img_ids), ", ") * ")"
    println(cmdline)
    ds = DBInterface.execute(db, cmdline) |> DataFrame
    data = Dict()
    println("Coord", ds.coord)
    coords = hcat(map(crd -> reinterpret(Float64, crd), ds.coord)...)

    gradients = hcat(map(crd -> reinterpret(Float64, crd), ds.gradients)...)
    # N x D
    data["coords"] = permutedims(coords, [2, 1])
    data["gradients"] = permutedims(gradients, [2, 1])
    data["potential"] = Array{Float64, 1}(reinterpret(Float64, ds.potential))
    println("potential", data["potential"])
    println("gradients", data["gradients"])
    return data
end


end
