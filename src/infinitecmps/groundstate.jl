function groundstate(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL - (α/2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dRs = d
        return dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return E, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return finalize!(x, E, d, numiter)
    end

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
    #return optimtest(fg, x; retract = retract, inner = inner)
end

function groundstate2(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = LBFGS(20; verbosity = 2),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL - (α/2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dRs = d
        return dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return E, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return finalize!(x, E, d, numiter)
    end

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end

function groundstate3(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        Ψ, = x
        Q = Ψ.Q
        Rs = Ψ.Rs

        dQ, dRs = d

        Rs = Rs .+ α .* dRs
        Q = Q + α * dQ

        Ψ = InfiniteCMPS(Q, Rs)
        ρR, λ, infoR = rightenv(Ψ; eigalg = eigalg, linalg = linalg, kwargs...)
        #@show λ
        #ρL, λ, infoR = leftenv(Ψ; eigalg = eigalg, linalg = linalg, kwargs...)
        #@show λ
        ρL, ρR, infoL, infoR = environments!(Ψ; eigalg = eigalg, linalg = linalg, kwargs...)
        #rmul!(ρR, 1/tr(ρR[]))
        #rmul!(ρL, 1/tr(ρL[]))
        HL, E, e, hL, infoL = leftenv(H, (Ψ,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)
        HR, E, e, hR, infoR = rightenv(H, (Ψ,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        dQ = dQ - (λ/α)*one(Q)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (Ψ, ρL, ρR, HL, HR, E, e, hL, hR), (dQ,dRs)
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        gradQ1, gradRs1 = d1
        gradQ2, gradRs2 = d2
        s = gradQ1 === gradQ2 ? 2*norm(gradQ1)^2 : 2*real(dot(gradQ1, gradQ2))
        for (gradR1,gradR2) in zip(gradRs1, gradRs2)
            if gradR1 === gradR2
                s += 2*norm(gradR1)^2
            else
                s += 2*real(dot(gradR1, gradR2))
            end
        end
        return s
    end

    function precondition(x, d)
        Ψ, ρL, ρR, = x
        dQ, dRs = d
        #dQ = posreginv(ρL[0], δ) * dQ * posreginv(ρR[0], δ)
        dRs = Ref(posreginv(ρL[0], δ)) .* dRs .* Ref(posreginv(ρR[0], δ))
        return (dQ, dRs)
    end

    function fg(x)
        (Ψ, ρL, ρR, HL, HR, E, e, hL, hR) = x

        gradQ, gradRs = gradient(H, (Ψ, ρL, ρR), HL, HR; kwargs...)

        Q = Ψ.Q
        D = Int(sqrt(size(Q[])[1]))
        Rs = Ψ.Rs

        Id = 1*Matrix(I,D,D)
        gradR1 = gradRs[1][]
        gradR2 = gradRs[2][]
        gradR1 = reshape(gradR1,D,D,D,D)
        gradR2 = reshape(gradR2,D,D,D,D)
        @tensor gradR1[a,b] := gradR1[a,c,b,c]
        @tensor gradR2[a,b] := gradR2[c,a,c,b]
        gradR1 = Constant(kron(Id,gradR1/D))
        gradR2 = Constant(kron(gradR2/D,Id))
        gradRs = (gradR1,gradR2)

        return E, (gradQ, gradRs)
    end
    function scale!(d, α)
        dQ, dRs = d
        rmul!(dQ, α)
        for dR in dRs
            rmul!(dR, α)
        end
        return d
    end
    function add!(d1, d2, α)
        dQ1, dR1s = d1
        dQ2, dR2s = d2
        axpy!(α, dQ2, dQ1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(α, dR2, dR1)
        end
        return d1
    end
    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        @show normgrad2
        δ = max(1e-12, 1e-3*normgrad2)
        return finalize!(x, E, d, numiter)
    end
    ρL, ρR, infoL, infoR = environments!(Ψ₀; eigalg = eigalg, linalg = linalg, kwargs...)
    #rmul!(ρR, 1/tr(ρR[]))
    #rmul!(ρL, 1/tr(ρL[]))
    HL, E, e, hL, infoL = leftenv(H, (Ψ₀,ρL,ρR); kwargs...)
    HR, E, e, hR, infoL = rightenv(H, (Ψ₀,ρL,ρR); kwargs...)
    x = (Ψ₀, ρL, ρR, HL, HR, E, e, hL, hR)

    x, E, normgrad, numfg, history = optimize(fg, x, optalg; retract = retract,
    #                            precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)

    (Ψ, ρL, ρR, HL, HR, E, e, hL, hR) = x
    return Ψ, ρL, ρR, E, e, normgrad, numfg, history
    #return optimtest(fg, x; retract = retract, inner = inner)
end

function groundstate4(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dK, dRs = d

        RLs = RLs .+ α .* dRs
        KL = KL + α * dK
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dK1, dRs1 = d1
        dK2, dRs2 = d2
        s = dK1 === dK2 ? 2*norm(dK1)^2 : 2*real(dot(dK1, dK2))
        for (dRs1,dRs2) in zip(dRs1, dRs2)
            if dRs1 === dRs2
                s += 2*norm(dRs1)^2
            else
                s += 2*real(dot(dRs1, dRs2))
            end
        end
        return s
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dK, dRs = d

        Q = ΨL.Q
        Rs = ΨL.Rs
        D = Int(sqrt(size(Q[])[1]))

        Id = 1*Matrix(I,D,D)
        ρR1 = reshape(ρR[0],D,D,D,D)
        ρR2 = reshape(ρR[0],D,D,D,D)
        @tensor ρR1[a,b] := ρR1[a,c,b,c]
        @tensor ρR2[a,b] := ρR2[c,a,c,b]

        dR1 = dRs[1][]
        dR2 = dRs[2][]
        dR1 = reshape(dR1,D,D,D,D)
        dR2 = reshape(dR2,D,D,D,D)
        @tensor dR1[a,b] := dR1[a,c,b,c]
        @tensor dR2[a,b] := dR2[c,a,c,b]
        dR1 = Constant(kron(Id,dR1*posreginv(ρR1, δ)/D))
        dR2 = Constant(kron(dR2*posreginv(ρR2, δ)/D,Id))
        dRs = (dR1,dR2)

        return (dK,dRs)
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Q = ΨL.Q
        Rs = ΨL.Rs
        D = Int(sqrt(size(Q[])[1]))

        # Id = 1*Matrix(I,D,D)
        # gradR1 = gradRs[1][]
        # gradR2 = gradRs[2][]
        # gradR1 = reshape(gradR1,D,D,D,D)
        # gradR2 = reshape(gradR2,D,D,D,D)
        # @tensor gradR1[a,b] := gradR1[a,c,b,c]
        # @tensor gradR2[a,b] := gradR2[c,a,c,b]
        # gradR1 = Constant(kron(Id,gradR1/D))
        # gradR2 = Constant(kron(gradR2/D,Id))
        # gradRs = (gradR1,gradR2)

        dK = 0.5*(gradQ - gradQ')
        dRs = gradRs .- (Rs) .* Ref(0.5*(gradQ + gradQ'))

        Id = 1*Matrix(I,D,D)
        dR1 = dRs[1][]
        dR2 = dRs[2][]
        dR1 = reshape(dR1,D,D,D,D)
        dR2 = reshape(dR2,D,D,D,D)
        @tensor dR1[a,b] := dR1[a,c,b,c]
        @tensor dR2[a,b] := dR2[c,a,c,b]
        dR1 = Constant(kron(Id,dR1/D))
        dR2 = Constant(kron(dR2/D,Id))
        dRs = (dR1,dR2)

        return E, (dK, dRs)
    end

    function scale!(d, α)
        dK, dRs = d
        rmul!(dK, α)
        for dR in dRs
            rmul!(dR, α)
        end
        return d
    end
    function add!(d1, d2, α)
        dK1, dR1s = d1
        dK2, dR2s = d2
        axpy!(α, dK2, dK1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(α, dR2, dR1)
        end
        return d1
    end

    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return finalize!(x, E, d, numiter)
    end

    ΨL₀ = Ψ₀
    #ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
    #                            finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
    #return optimtest(fg, x; retract = retract, inner = inner)
end
