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
        ρL, ρR, infoL, infoR = environments!(Ψ; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        rmul!(ρL, 1/tr(ρL[]))
        HL, E, e, hL, infoL = leftenv(H, (Ψ,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)
        HR, E, e, hR, infoR = rightenv(H, (Ψ,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)


        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (Ψ, ρL, ρR, HL, HR, E, e, hL, hR), d
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
        return posreginv(ρL[0], δ) * dQ * posreginv(ρR[0], δ), Ref(posreginv(ρL[0], δ)) .* dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        (Ψ, ρL, ρR, HL, HR, E, e, hL, hR) = x

        gradQ, gradRs = gradient_tens_prod(H, (Ψ, ρL, ρR), HL, HR; kwargs...)

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
    rmul!(ρR, 1/tr(ρR[]))
    rmul!(ρL, 1/tr(ρL[]))
    HL, E, e, hL, infoL = leftenv(H, (Ψ₀,ρL,ρR); kwargs...)
    HR, E, e, hR, infoL = rightenv(H, (Ψ₀,ρL,ρR); kwargs...)
    x = (Ψ₀, ρL, ρR, HL, HR, E, e, hL, hR)

    #x, E, normgrad, numfg, history = optimize(fg, x, optalg; retract = retract,
                                #precondition = precondition,
                                #finalize! = _finalize!,
                                #inner = inner, transport! = transport!,
                                #scale! = scale!, add! = add!,
                                #isometrictransport = true)

    #(Ψ, ρL, ρR, HL, HR, E, e, hL, hR) = x
    #return Ψ, ρL, ρR, E, e, normgrad, numfg, history
    return optimtest(fg, x; retract = retract, inner = inner)
end
