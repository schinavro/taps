module LBFGSBMPI

using Base
using LBFGSB
using MPI

export L_BFGS_B

struct L_BFGS_B
    nmax::Int
    mmax::Int
    task::Vector{UInt8}
    csave::Vector{UInt8}
    lsave::Vector{Cint}
    isave::Vector{Cint}
    dsave::Vector{Cdouble}
    wa::Vector{Cdouble}
    iwa::Vector{Cint}
    g::Vector{Cdouble}
    nbd::Vector{Cint}
    l::Vector{Cdouble}
    u::Vector{Cdouble}
    function L_BFGS_B(nmax, mmax)
        task = fill(Cuchar(' '), 60)
        csave = fill(Cuchar(' '), 60)
        lsave = zeros(Cint, 4)
        isave = zeros(Cint, 44)
        dsave = zeros(Cdouble, 29)
        wa = zeros(Cdouble, 2mmax*nmax + 5nmax + 11mmax*mmax + 8mmax)
        iwa = zeros(Cint, 3*nmax)
        g = zeros(Cdouble, nmax)
        nbd = zeros(Cint, nmax)
        l = zeros(Cdouble, nmax)
        u = zeros(Cdouble, nmax)
        new(nmax, mmax, task, csave, lsave, isave, dsave, wa, iwa, g, nbd, l, u)
    end
end

using StringEncodings

function (obj::L_BFGS_B)(𝒮, ∂𝒮, x0, comm;
                         m=10, factr=1e7, pgtol=1e-5, iprint=-1, maxfun=200,
                         maxiter=100)
    Nk, D = Base.size(x0)
    x = vec(x0)
    n = Nk * D
    f = 0.0
    # clean up
    fill!(obj.task, Cuchar(' '))
    fill!(obj.csave, Cuchar(' '))
    fill!(obj.lsave, zero(Cint))
    fill!(obj.isave, zero(Cint))
    fill!(obj.dsave, zero(Cdouble))
    fill!(obj.wa, zero(Cdouble))
    fill!(obj.iwa, zero(Cint))
    fill!(obj.g, zero(Cdouble))
    fill!(obj.nbd, zero(Cint))
    fill!(obj.l, zero(Cdouble))
    fill!(obj.u, zero(Cdouble))
    # set bounds
    for i = 1:n
        obj.nbd[i] = 0
        obj.l[i] = -Inf
        obj.u[i] = Inf
    end
    # start
    obj.task[1:5] = b"START"
    nprc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    root = 0

    while true
        if rank == root
            setulb(n, m, x, obj.l, obj.u, obj.nbd, f, obj.g, factr, pgtol, obj.wa,
                   obj.iwa, obj.task, iprint, obj.csave, obj.lsave, obj.isave, obj.dsave)
            println(decode(obj.task, "UTF-8"))

        end
        MPI.Bcast!(x, root, comm)
        MPI.Bcast!(obj.task, root, comm)

        if obj.task[1:2] == b"FG"
            f = 𝒮(reshape(x, Nk, D))
            ∂𝒮(obj.g, reshape(x, Nk, D))
        elseif obj.task[1:5] == b"NEW_X"
            MPI.Bcast!(obj.isave, root, comm)
            if obj.isave[30] ≥ maxiter
                obj.task[1:43] = b"STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
                return f, reshape(x, Nk, D)
            elseif obj.isave[34] ≥ maxfun
                obj.task[1:52] = b"STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
                return f, reshape(x, Nk, D)
            end
        else
            println(obj.dsave)
            return f, reshape(x, Nk, D)
        end
    end
end

end
