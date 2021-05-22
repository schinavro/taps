using JSON
using DataStructures: Deque
#Taps = include("/home/schinavro/libCalc/taps/taps_parallel/taps.jl")
#using .Taps: Coords

int, float, real, number, bool = 1, 2, 3, 4, 5
int8, int16, int32, int64 = 6, 7, 8, 9
uint8, uint16, uint32, uint64 = 10, 11, 12, 13
float16, float32, float64 = 14, 15, 16
complex64, complex128 = 17, 18

const Float = AbstractFloat

typemaps = Dict(
    int=>Int, Int=>int,
    float=>Float, Float=>float, Real=>float, Number=>float,
    bool=>Bool, Bool=>bool,
    int8=>Int8, Int8=>int8,
    int16=>Int16, Int16=>int16,
    int32=>Int32, Int32=>int32,
    int64=>Int64, Int64=>int64,
    uint8=>UInt8, UInt8=>uint8,
    uint16=>UInt16, UInt16=>uint16,
    uint32=>UInt32, UInt32=>uint32,
    uint64=>UInt64, UInt64=>uint64,
    float16=>Float16, Float16=>float16,
    float32=>Float32, Float32=>float32,
    float64=>Float64, Float64=>float64,
    complex64=>Complex{Float32}, Complex{Float32}=>complex64,
    complex128=>Complex{Float64}, Complex{Float64}=>complex128
)
ordermaps = Dict("C"=>0, 0=>"C", 1=>"F", "F"=>1)

function write_header(pointer, arr)
    header = []
    dtype = eltype(arr)
    ndim = ndims(arr)
    shape = size(arr)
    order = "F"
    header = Array{Int64, 1}([pointer, typemaps[dtype], ndim, shape..., ordermaps[order]])
    headerbytes = reinterpret(UInt8, header)
    return headerbytes
end

function read_header(arrbytes)
    header_size = reinterpret(Int64, arrbytes[1:8])[1]
    header = reinterpret(Int64, arrbytes[9:8+header_size])
    pointer, dtype, ndim = header[1], typemaps[header[2]], header[3]
    shape = header[4:3+ndim]
    order = ordermaps[header[4+ndim]]
    header = header_size, pointer, dtype, ndim, shape, order
    return header
end

function statify(arrlist, d::Union{Dict, Array{Any, 1}})
    pointerlist = ["__$(arr[1])__"  for arr in arrlist]
    global queue = Deque{Tuple}()
    push!(queue, (hash(d), d))
    memo = Set()
    while true
        global queue
        isempty(queue) ? break : nothing
        id_, o = popfirst!(queue)
        id_ in memo ? continue : nothing
        union!(memo, id_)

        if typeof(o)<:Dict for (k, v) in pairs(o)
            push!(queue, (hash(v), v))
            v in pointerlist ? o[k] = arrlist[findfirst(x->x==v, pointerlist)][2] : nothing
            end
        elseif typeof(o)<:Union{Array{Any, 1}, Tuple} for i=1:length(o)
            v = o[i]
            push!(queue, (hash(v), v))
            v in pointerlist ? o[i] = arrlist[findfirst(x->x==v, pointerlist)][2] : nothing
            end
        end
    end
    return d
end

function pointify(d, pointer, binarylist::Array)
    global queue = Deque{Tuple}()
    push!(queue, (hash(d), d))
    memo = Set()
    while true
        global queue
        isempty(queue) ? break : nothing
        id_, o = popfirst!(queue)
        id_ in memo ? continue : nothing
        union!(memo, id_)
        if typeof(o)<:Dict for (k, v) in pairs(o)
            push!(queue, ((hash(v), v)))
            if typeof(v)<:Array{T, N} where {T<:Number, N}
                pstr = "__$(pointer)__"

                headerbytes = write_header(pointer, o[k])
                binar = cat(reinterpret(UInt8, [length(headerbytes)]), headerbytes, reinterpret(UInt8, vec(v)), dims=1)
                push!(binarylist, binar)
                o[k] = pstr
                pointer += 1
            end end
        elseif typeof(o)<:Union{Array{Any, 1}, Tuple} for i=1:length(o)
            v = o[i]
            push!(queue, (hash(v), v))
            if typeof(v)<:Array{T, N} where {T<:Number, N}

                pstr = "__$(pointer)__"
                headerbytes = write_header(pointer, v)
                binar = cat(reinterpret(UInt8, [length(headerbytes)]), headerbytes, reinterpret(UInt8, vec(v)), dims=1)
                push!(binarylist, binar)
                o[i] = pstr
                pointer += 1
            end end
        end
    end
    return d, pointer, binarylist
end

function packing(args...; kwargs...)::Array{UInt8, 1}
    args, kwargs = Array{Any, 1}([args...]), Dict{Symbol, Any}(kwargs)
    binarylist = []
    pointer = 0

    args, pointer, binarylist = pointify(args, pointer, binarylist)
    kwargs, pointer, binarylist = pointify(kwargs, pointer, binarylist)
    kwargs = Dict{Any, Any}(kwargs)

    kwargs[:args] = args

    howmanybinary = binarylist == [] ? 0 : reinterpret(UInt8, [length(binarylist)])

    eachsize = binarylist == [] ? [] : reinterpret(UInt8, [length(b) for b in binarylist])

    binarybytes = cat(howmanybinary, eachsize, cat(binarylist..., dims=1), dims=1)
    kwargsbytes = Array{UInt8}(JSON.json(kwargs))

    data = cat(binarybytes, kwargsbytes, dims=1)
    header = reinterpret(UInt8, [length(data)+16, length(binarybytes), length(kwargsbytes)])
    return cat(header, data, dims=1)
end


function unpacking(bytesarr::Array{UInt8, 1})
    binarysize, kwargssize = reinterpret(Int64, bytesarr[1:16])
    partition = 16 + binarysize

    binarybytes = bytesarr[17:partition]
    kwargsbytes = bytesarr[partition+1:end]

    #binarybytes to binarylist
    howmanybinary = reinterpret(Int64, binarybytes[1:8])[1]
    eachsize = reinterpret(Int64, binarybytes[9:8 + 8howmanybinary])
    partition = 9 + 8howmanybinary

    arrlist = []
    for size in eachsize
        arrbytes = binarybytes[partition:partition+size-1]
        header_size, pointer, dtype, ndim, shape, order = read_header(arrbytes)

        shape = order == "C" ? reverse(shape) : shape

        arr = reinterpret(dtype, arrbytes[header_size+9:end])

        array = Array{dtype, length(shape)}(reshape(arr, shape...))
        push!(arrlist, (pointer, array))
        partition += size
    end

    pkwargs = JSON.parse(String(kwargsbytes); dicttype=Dict{Symbol,Any})
    pargs = pkwargs[:args]
    delete!(pkwargs, :args)

    # link binarybytes and everything
    args = statify(arrlist, pargs)
    kwargs = statify(arrlist, pkwargs)

    return args, kwargs
end
