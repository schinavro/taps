using ForwardDiff, SparseArrays
abstract type Hyperparameters end;

struct KernelHyperparameters <: Hyperparameters
    array
end

function kernel(sig::Number, l::Number, rcut::Number, ri::Vector, rj::Vector,
                ri1::Vector, rj1::Vector)
    rii1 = norm(ri .- ri1)
    rjj1 = norm(rj .- rj1)

    dij = abs(rii1-rjj1)

    if rcut < dij
        return 0.
    end
    return sig * exp(-(dij)^2/(2*l^2)) * (rcut - dij)^2
end

function kernel(sig::Number, l::Number, rcut::Number, ri::Vector, rj::Vector,
                ri1::Vector, rj1::Vector, ri2::Vector, rj2::Vector)

    rii1 = norm(ri .- ri1)
    rii2 = norm(ri .- ri2)
    ri1i2 = norm(ri1 .- ri2)

    rjj1 = norm(rj .- rj1)
    rjj2 = norm(rj .- rj2)
    rj1j2 = norm(rj1 .- rj2)

    dij = sqrt((rii1-rjj1)^2 + (rii2-rjj2)^2 + (ri1i2-rj1j2)^2)

    if rcut < dij
        return 0.
    end
    return sig * exp(-(dij)^2/(2*l^2)) * (rcut - dij)^2
end

function kgd(sig::Number, l::Number, rcut::Number, ri::Vector, rj::Vector, ri1::Vector, rj1::Vector)
    kgg(rri::Vector, rrj::Vector) = kernel(hyperparameters.array..., rcut, rri, rrj, ri1, rj1)
    kgd(rri::Vector, rrj::Vector) = ForwardDiff.gradient(x->kgg(x, rrj), rri)
    return kgd(ri, rj)
end

function kgd(sig::Number, l::Number, rcut::Number, ri::Vector, rj::Vector, ri1::Vector, rj1::Vector, ri2::Vector, rj2::Vector)
    kgg(rri::Vector, rrj::Vector) = kernel(hyperparameters.array..., rcut, rri, rrj, ri1, rj1, ri2, rj2)
    kgd(rri::Vector, rrj::Vector) = ForwardDiff.gradient(x->kgg(x, rrj), rri)
    return kgd(ri, rj)
end

function kgd(hyperparameters, rcut, ri::Vector, rj::Vector, ri1::Vector, ri2::Vector, rj1::Vector, rj2::Vector)
    kgg(rri::Vector, rrj::Vector) = kernel(hyperparameters.array..., rcut, rri, rrj, ri1, rj1, ri2, rj2)
    kgd(rri::Vector, rrj::Vector) = ForwardDiff.gradient(x->kgg(x, rrj), rri)
    return kgd(ri, rj)
end

function kdg(hyperparameters, rcut, ri::Vector, rj::Vector, ri1::Vector, ri2::Vector, rj1::Vector, rj2::Vector)
    kgg(rri::Vector, rrj::Vector) = kernel(hyperparameters.array..., rcut, rri, rrj, ri1, rj1, ri2, rj2)
    kgd(rri::Vector, rrj::Vector) = ForwardDiff.gradient(x->kgg(rri, x), rrj)
    return kgd(ri, rj)
end

function kdd(hyperparameters, rcut, ri::Vector, rj::Vector, ri1::Vector, ri2::Vector, rj1::Vector, rj2::Vector)
    kgg(rri::Vector, rrj::Vector) = kernel(hyperparameters.array..., rcut, rri, rrj, ri1, rj1, ri2, rj2)
    kgd(rri::Vector, rrj::Vector) = ForwardDiff.gradient(x->kgg(x, rrj), rri)
    kdd(rri::Vector, rrj::Vector) = ForwardDiff.jacobian(x->kgd(rri, x), rrj)
    return kdd(ri, rj)
end

function khd(hyperparameters, rcut, ri::Vector, rj::Vector, ri1::Vector, ri2::Vector, rj1::Vector, rj2::Vector)
    kgg(rri::Vector, rrj::Vector) = kernel(hyperparameters.array..., rcut, rri, rrj, ri1, rj1, ri2, rj2)
    kgd(rri::Vector, rrj::Vector) = ForwardDiff.gradient(x->kgg(x, rrj), rri)
    kdd(rri::Vector, rrj::Vector) = ForwardDiff.jacobian(x->kgd(rri, x), rrj)
    khd(rri::Vector, rrj::Vector) = ForwardDiff.jacobian(x->kdd(x, rrj), rri)
    return khd(ri, rj)
end


function kgd(descriptor::Taps.TwoThreeBody, db::Taps.TwoThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    K2 = kgd(descriptor, db.threebody, hyperparameters, rcut)
    K3 = kgd(descriptor, db.threebody, hyperparameters, rcut)
    return K2 + K3
end

function kdg(descriptor::Taps.TwoThreeBody, db::Taps.TwoThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    K2 = kdg(descriptor, db.threebody, hyperparameters, rcut)
    K3 = kdg(descriptor, db.threebody, hyperparameters, rcut)
    return K2 + K3
end

function kdd(descriptor::Taps.TwoThreeBody, db::Taps.TwoThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    K2 = kdd(descriptor, db.threebody, hyperparameters, rcut)
    K3 = kdd(descriptor, db.threebody, hyperparameters, rcut)
    return K2 + K3
end

function khd(descriptor::Taps.TwoThreeBody, db::Taps.TwoThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    K2 = khd(descriptor, db.threebody, hyperparameters, rcut)
    K3 = khd(descriptor, db.threebody, hyperparameters, rcut)
    return K2 + K3
end


function kgg(sparsetwothreebody::Taps.SparseAtomic{T, NN}, db::Taps.TwoThreeBodyDB,
             hyperparameters::KernelHyperparameters, rcut) where {T<:Taps.TwoThreeBody, NN}
    coords = sparsetwothreebody.coords
    N, A = size(coords)

    K = zeros(N*A, db.m)

    count = 1
    for i=1:N
        for a=1:A
            K[count, :] = kgg(coords[i, a], db, hyperparamters, rcut)
            count += 1
        end
    end
    return K
end

function kgg(descriptor::Taps.TwoThreeBody, db::Taps.TwoThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    K2 = kgg(descriptor, db.threebody, hyperparameters, rcut)
    K3 = kgg(descriptor, db.threebody, hyperparameters, rcut)
    return K2 + K3
end


function kgg(descriptor::Taps.TwoThreeBody, db::Taps.TwoBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Ker = spzeros(maxidx)
    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        K = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]

                for permute in permutations
                    rj, rj1 = [rrj, rrj1][permute]
                    K[idx] += kernel(hyperparameters.array..., rcut, ri, rj, ri1, rj1)
                end
            end
        end
        Ker .+= K
    end
    return Ker
end


function kgg(descriptor::Taps.TwoThreeBody, db::Taps.ThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Ker = spzeros(maxidx)
    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        K = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]
            ri2 = [dscpt[cluster]["xi2"][ci], dscpt[cluster]["yi2"][ci], dscpt[cluster]["zi2"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]
                rrj2 = [db.xi2[cluster][cj], db.yi2[cluster][cj], db.zi2[cluster][cj]]

                for permute in permutations
                    rj, rj1, rj2 = [rrj, rrj1, rrj2][permute]
                    K[idx] += kernel(hyperparameters.array..., rcut, ri, rj, ri1, rj1, ri2, rj2)
                end
            end
        end
        Ker .+= K
    end
    return Ker
end

function kgd(descriptor::Taps.TwoThreeBody, db::Taps.TwoBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Kerx = spzeros(maxidx)
    Kery = spzeros(maxidx)
    Kerz = spzeros(maxidx)
    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        Kx = spzeros(maxidx)
        Ky = spzeros(maxidx)
        Kz = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]

                for permute in permutations
                    rj, rj1 = [rrj, rrj1][permute]
                    dK = kgd(hyperparameters, rcut, ri, rj, ri1, rj1)
                    Kx[idx] += dK[1]
                    Ky[idx] += dK[2]
                    Kz[idx] += dK[3]
                end
            end
        end
        Kerx .+= Kx
        Kery .+= Ky
        Kerz .+= Kz
    end
    return [Kerx, Kery, Kerz]
end


function kgd(descriptor::Taps.TwoThreeBody, db::Taps.ThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Kerx = spzeros(maxidx)
    Kery = spzeros(maxidx)
    Kerz = spzeros(maxidx)

    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        Kx = spzeros(maxidx)
        Ky = spzeros(maxidx)
        Kz = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]
            ri2 = [dscpt[cluster]["xi2"][ci], dscpt[cluster]["yi2"][ci], dscpt[cluster]["zi2"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]
                rrj2 = [db.xi2[cluster][cj], db.yi2[cluster][cj], db.zi2[cluster][cj]]

                for permute in permutations
                    rj, rj1, rj2 = [rrj, rrj1, rrj2][permute]
                    dK = kgd(hyperparameters, rcut, ri, rj, ri1, rj1, ri2, rj2)
                    Kx[idx] += dK[1]
                    Ky[idx] += dK[2]
                    Kz[idx] += dK[3]
                end
            end
        end
        Kerx .+= Kx
        Kery .+= Ky
        Kerz .+= Kz
    end
    return [Kerx, Kery, Kerz]
end


function kdg(descriptor::Taps.TwoThreeBody, db::Taps.TwoBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Kerx = spzeros(maxidx)
    Kery = spzeros(maxidx)
    Kerz = spzeros(maxidx)
    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        Kx = spzeros(maxidx)
        Ky = spzeros(maxidx)
        Kz = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]

                for permute in permutations
                    rj, rj1 = [rrj, rrj1][permute]
                    dK = kgd(hyperparameters, rcut, ri, rj, ri1, rj1)
                    Kx[idx] += dK[1]
                    Ky[idx] += dK[2]
                    Kz[idx] += dK[3]
                end
            end
        end
        Kerx .+= Kx
        Kery .+= Ky
        Kerz .+= Kz
    end
    return [Kerx, Kery, Kerz]
end


function kdg(descriptor::Taps.TwoThreeBody, db::Taps.ThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Kerx = spzeros(maxidx)
    Kery = spzeros(maxidx)
    Kerz = spzeros(maxidx)

    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        Kx = spzeros(maxidx)
        Ky = spzeros(maxidx)
        Kz = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]
            ri2 = [dscpt[cluster]["xi2"][ci], dscpt[cluster]["yi2"][ci], dscpt[cluster]["zi2"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]
                rrj2 = [db.xi2[cluster][cj], db.yi2[cluster][cj], db.zi2[cluster][cj]]

                for permute in permutations
                    rj, rj1, rj2 = [rrj, rrj1, rrj2][permute]
                    dK = kdg(hyperparameters, rcut, ri, rj, ri1, rj1, ri2, rj2)
                    Kx[idx] += dK[1]
                    Ky[idx] += dK[2]
                    Kz[idx] += dK[3]
                end
            end
        end
        Kerx .+= Kx
        Kery .+= Ky
        Kerz .+= Kz
    end
    return [Kerx, Kery, Kerz]
end

function kdd(descriptor::Taps.TwoThreeBody, db::Taps.TwoBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Kerxx = spzeros(maxidx)
    Kerxy = spzeros(maxidx)
    Kerxz = spzeros(maxidx)

    Keryy = spzeros(maxidx)
    Keryx = spzeros(maxidx)
    Keryz = spzeros(maxidx)

    Kerzx = spzeros(maxidx)
    Kerzy = spzeros(maxidx)
    Kerzz = spzeros(maxidx)
    for cluster in chart
        permutations = dscpt[cluster]["permutations"]
        Kxx = spzeros(maxidx)
        Kxy = spzeros(maxidx)
        Kxz = spzeros(maxidx)

        Kyy = spzeros(maxidx)
        Kyx = spzeros(maxidx)
        Kyz = spzeros(maxidx)

        Kzx = spzeros(maxidx)
        Kzy = spzeros(maxidx)
        Kzz = spzeros(maxidx)
        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]

                for permute in permutations
                    rj, rj1 = [rrj, rrj1][permute]
                    dK = kdd(hyperparameters, rcut, ri, rj, ri1, rj1)
                    Kxx[idx] += dK[1, 1]
                    Kxy[idx] += dK[1, 2]
                    Kxz[idx] += dK[1, 3]

                    Kyx[idx] += dK[2, 1]
                    Kyy[idx] += dK[2, 2]
                    Kyz[idx] += dK[2, 3]

                    Kzx[idx] += dK[3, 1]
                    Kzy[idx] += dK[3, 2]
                    Kzz[idx] += dK[3, 3]
                end
            end
        end
        Kerxx .+= Kxx
        Kerxy .+= Kxy
        Kerxz .+= Kxz

        Keryx .+= Kyx
        Keryy .+= Kyy
        Keryz .+= Kyz

        Kerzx .+= Kzx
        Kerzy .+= Kzy
        Kerzz .+= Kzz
    end
    return [Kerxx, Kerxy, Kerxz, Keryx, Keryy, Keryz, Kerzx, Kerzy, Kerzz]
end

function kdd(descriptor::Taps.TwoThreeBody, db::Taps.ThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    Kerxx = spzeros(maxidx)
    Kerxy = spzeros(maxidx)
    Kerxz = spzeros(maxidx)

    Keryy = spzeros(maxidx)
    Keryx = spzeros(maxidx)
    Keryz = spzeros(maxidx)

    Kerzx = spzeros(maxidx)
    Kerzy = spzeros(maxidx)
    Kerzz = spzeros(maxidx)

    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        Kxx = spzeros(maxidx)
        Kxy = spzeros(maxidx)
        Kxz = spzeros(maxidx)

        Kyy = spzeros(maxidx)
        Kyx = spzeros(maxidx)
        Kyz = spzeros(maxidx)

        Kzx = spzeros(maxidx)
        Kzy = spzeros(maxidx)
        Kzz = spzeros(maxidx)

        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]
            ri2 = [dscpt[cluster]["xi2"][ci], dscpt[cluster]["yi2"][ci], dscpt[cluster]["zi2"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]
                rrj2 = [db.xi2[cluster][cj], db.yi2[cluster][cj], db.zi2[cluster][cj]]

                for permute in permutations
                    rj, rj1, rj2 = [rrj, rrj1, rrj2][permute]
                    dK = kdd(hyperparameters, rcut, ri, rj, ri1, rj1, ri2, rj2)

                    Kxx[idx] += dK[1, 1]
                    Kxy[idx] += dK[1, 2]
                    Kxz[idx] += dK[1, 3]

                    Kyx[idx] += dK[2, 1]
                    Kyy[idx] += dK[2, 2]
                    Kyz[idx] += dK[2, 3]

                    Kzx[idx] += dK[3, 1]
                    Kzy[idx] += dK[3, 2]
                    Kzz[idx] += dK[3, 3]
                end
            end
        end
        Kerxx .+= Kxx
        Kerxy .+= Kxy
        Kerxz .+= Kxz

        Keryx .+= Kyx
        Keryy .+= Kyy
        Keryz .+= Kyz

        Kerzx .+= Kzx
        Kerzy .+= Kzy
        Kerzz .+= Kzz
    end
    return [Kerxx, Kerxy, Kerxz, Keryx, Keryy, Keryz, Kerzx, Kerzy, Kerzz]
end

function khd(descriptor::Taps.TwoThreeBody, db::Taps.TwoBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    # 3xJxK
    Kerijk = [spzeros(maxidx) for ii=1:27]

    for cluster in chart
        permutations = dscpt[cluster]["permutations"]
        Kijk = [spzeros(maxidx) for ii=1:27]

        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]

                for permute in permutations
                    rj, rj1 = [rrj, rrj1][permute]
                    dK = kdd(hyperparameters, rcut, ri, rj, ri1, rj1)


                    for ii=1:9
                        Kijk[ii][idx] += dK[ii, 1]
                        Kijk[ii+9][idx] += dK[ii, 2]
                        Kijk[ii+18][idx] += dK[ii, 3]
                    end
                end
            end
        end
        for ii=1:27
            Kerijk[ii] .+= Kijk[ii]
        end
    end
    return Kerijk
end

function khd(descriptor::Taps.TwoThreeBody, db::Taps.ThreeBodyDB, hyperparameters::KernelHyperparameters, rcut)
    dscpt = descriptor.descriptor
    chart = keys(db.idx)

    maxidx = 0
    for cluster in chart
        length(db.idx[cluster]) == 0 ? continue : nothing
        maxidx = max(maxidx, db.idx[cluster][end])
    end

    # 3xJxK
    Kerijk = [spzeros(maxidx) for ii=1:27]

    for cluster in chart

        permutations = dscpt[cluster]["permutations"]
        Kijk = [spzeros(maxidx) for i=1:27]

        for ci=1:length(dscpt[cluster]["xi"])
            ri = [dscpt[cluster]["xi"][ci], dscpt[cluster]["yi"][ci], dscpt[cluster]["zi"][ci]]
            ri1 = [dscpt[cluster]["xi1"][ci], dscpt[cluster]["yi1"][ci], dscpt[cluster]["zi1"][ci]]
            ri2 = [dscpt[cluster]["xi2"][ci], dscpt[cluster]["yi2"][ci], dscpt[cluster]["zi2"][ci]]

            for cj=1:length(db.idx[cluster])
                idx = db.idx[cluster][cj]

                rrj = [db.xi[cluster][cj], db.yi[cluster][cj], db.zi[cluster][cj]]
                rrj1 = [db.xi1[cluster][cj], db.yi1[cluster][cj], db.zi1[cluster][cj]]
                rrj2 = [db.xi2[cluster][cj], db.yi2[cluster][cj], db.zi2[cluster][cj]]

                for permute in permutations
                    rj, rj1, rj2 = [rrj, rrj1, rrj2][permute]
                    dK = khd(hyperparameters, rcut, ri, rj, ri1, rj1, ri2, rj2)

                    for ii=1:9
                        Kijk[ii][idx] += dK[ii, 1]
                        Kijk[ii+9][idx] += dK[ii, 2]
                        Kijk[ii+18][idx] += dK[ii, 3]
                    end
                end
            end
        end
        for ii=1:27
            Kerijk[ii] .+= Kijk[ii]
        end
    end
    return Kerijk
end
