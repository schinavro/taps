using ArgParse
parse_setting = ArgParseSettings()
@add_arg_table parse_setting begin
    "--io"
        help = "IO mode, `socket`, `file` default is `file`. "
        arg_type = String
        default = "file"
    "--host"
        help = "another option with an argument"
        arg_type = String
        default = "127.0.0.1"
    "--port"
        help = "another option with an argument"
        arg_type = Int
        default = 6543
    "--clienthost"
        help = "another option with an argument"
        arg_type = String
        default = "127.0.0.1"
    "--clientport"
        help = "another option with an argument"
        arg_type = Int
        default = 6544
    "--keep_open"
        help = "whether shutdown calculation after finish"
        action = :store_true
    "input_file"
        help = "a positional argument"
        required = false
end

args = parse_args(ARGS, parse_setting, as_symbols=true)

# Load MPI
import MPI
MPI.Init()
comm = MPI.COMM_WORLD
mpi_kwargs = Dict(:MPI => MPI, :comm => comm, :root => 0,
                  :nprc => MPI.Comm_size(comm), :rank => MPI.Comm_rank(comm),
                  :mFloat64 => MPI.Datatype(Float64),
                  :mUInt8 => MPI.Datatype(UInt8))

# Merge args and mpi
merge!(args, mpi_kwargs)

# Load TAPS IO module
if args[:io] == "file"
    IO = include("./file.jl")
    io = getfield(IO, :FileIO)(;args...)
elseif args[:io] == "socket"
    include("./socket.jl")
    # io = getfield(IO, :SocketIO)(;args...)
    io = SocketIO(;args...)
end

run_taps(io)
MPI.Finalize()
