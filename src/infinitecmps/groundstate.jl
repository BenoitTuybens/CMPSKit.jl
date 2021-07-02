using Printf

function _groundstate_constraint_infostring(ω, e, ns, μs = nothing, normgrad = nothing)
    s = @sprintf("ω = %.12f, e = %.12f", ω, e)
    N = length(ns)
    if N == 1
        s *= @sprintf(", n = %.6f", ns[1])
        if !isnothing(μs)
            s *= @sprintf(", μ = %.6f", μs[1])
        end
        if !isnothing(μs)
            s *= @sprintf(", ‖∇ω‖ = %.4e", normgrad)
        end
    else
        s *= ", ns = ("
        for k = 1:N
            s *= @sprintf("%.3f", ns[k])
            if k < N
                s *= ", "
            else
                s *= ")"
            end
        end
        if !isnothing(μs)
            s *= ", μs = ("
            for k = 1:N
                s *= @sprintf("%.3f", μs[k])
                if k < N
                    s *= ", "
                else
                    s *= ")"
                end
            end
        end
        if !isnothing(normgrad)
            s *= @sprintf("), ‖∇ω‖ = %.4e", normgrad)
        end
    end
    return s
end

groundstate4(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS; kwargs...) =
    groundstate4_unconstrained(Ĥ, Ψ₀; kwargs...)

    groundstate4(Ĥ::LocalHamiltonian,
                Ψ₀::UniformCMPS{<:AbstractMatrix,N},
                n₀s::NTuple{N,<:Number}; kwargs...) where {N} =
        groundstate4_constrained(Ĥ, Ψ₀, n₀s; kwargs...)

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

function groundstate4_unconstrained(H::LocalHamiltonian, Ψ₀::UniformCMPS;
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
    #                            precondition = precondition,
    #                            finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
    #return optimtest(fg, x; retract = retract, inner = inner)
end

function groundstate4_constrained(Ĥ::LocalHamiltonian,
                        Ψ₀::UniformCMPS{<:AbstractMatrix, N},
                        n₀s::NTuple{N,Number};
                        gradtol = 1e-7,
                        verbosity = 2,
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        chemical_potential_relaxation = 1.,
                        kwargs...) where {N}

    δ = 1
    μs = ntuple(k -> one(scalartype(Ψ₀)), N)
    n̂s = ntuple(k -> ψ[k]' * ψ[k],N)
    N̂s = ntuple(k -> ∫(n̂s[k], (-Inf,+Inf)), N)
    Ω̂ = Ĥ - sum( μs .* N̂s )
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dK, dRs, dμs = d

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
        ns = ntuple(k->expval(n̂s[k], ΨL, ρL, ρR)[], N)
        ΩL, Ω, ω, ωL, infoL =
            leftenv(Ω̂ , (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, ΩL, Ω, ω, ns, ωL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dK1, dRs1, dμs1 = d1
        dK2, dRs2, dμs2 = d2
        s = dK1 === dK2 ? 2*norm(dK1)^2 : 2*real(dot(dK1, dK2))
        for (dR1,dR2) in zip(dRs1, dRs2)
            if dR1 === dR2
                s += 2*norm(dR1)^2
            else
                s += 2*real(dot(dR1, dR2))
            end
        end
        s += sum(dμs1 .* dμs2)
        return s
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dK, dRs,  = d

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
        (ΨL, ρR, ΩL, Ω, ω, ωL) = x

        gradQ, gradRs = gradient(Ω̂ , (ΨL, one(ρR), ρR), ΩL, zero(ΩL); kwargs...)

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

        dμs = n₀s .- ns

        return E, (dK, dRs, dμs)
    end

    function scale!(d, α)
        dK, dRs, dμs = d
        rmul!(dK, α)
        for dR in dRs
            rmul!(dR, α)
        end
        for dμ in dμs
            dμ = dμ * α
        end
        return d
    end
    function add!(d1, d2, α)
        dK1, dR1s, dμs1 = d1
        dK2, dR2s, dμs2 = d2
        axpy!(α, dK2, dK1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(α, dR2, dR1)
        end
        for (dμ1, dμ2) in zip(dμs1,dμs2)
            dμ1 = α * dμ2 + dμ1
        end
        return d1
    end

    function _finalize!(x, E, d, numiter)
        (ΨL, ρR, ΩL, Ω, ω, ns, ωL) = x
        normgrad2 = real(inner(x, d, d))
        normgrad = sqrt(normgrad2)
        E = expval(density(Ĥ), ΨL, ρL, ρR)[]
        dμs = d[3]
        if verbosity > 1
            s = @sprintf("UniformCMPS ground state: iter %4d: ", numiter)
            s *= _groundstate_constraint_infostring(Ω, E, ns, μs, normgrad)
            @info s
        end
        μs = μs .+ chemical_potential_relaxation .* dμs
        Ω̂ = Ĥ - sum(μs .* N̂s)
        δ = max(1e-12, 1e-3*normgrad2)
        # recompute energy and gradient:
        ΩL, Ω, ω, ωL, info_ΩL =
            leftenv(Ω̂ , (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if info_ρR.converged == 0 || info_ΩL.converged == 0
            @warn "finalizing step with new chemical potential : not converged, ω = $Ω"
            @show info_ρR
            @show info_ΩL
        end

        x = (ΨL, ρR, ΩL, Ω, ω, ns, ωL)
        gradQ, gradRs = gradient(Ω̂ , (ΨL, ρL, ρR), ΩL, zero(ΩL); kwargs...)
        Rs = ΨL.Rs
        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs
        d = (dRs, dμs)
        return finalize!(x, Ω, d, numiter)
    end

    ΨL = Ψ₀
    #ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, info_ρR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    ns = ntuple(k->expval(n̂s[k], ΨL, ρL, ρR)[], N)
    # rescale initial cMPS to better approximate target densities, using geometric mean
    # this does not change the environments ρL and ρR
    scale_factor = prod( n₀s ./ ns)^(1/N)
    rmul!(ΨL.Q, scale_factor)
    rmul!.(ΨL.Rs, sqrt(scale_factor))
    ns = ns .* scale_factor
    ΩL, Ω, ω, ωL, info_ΩL =
        leftenv(Ω̂ , (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

    if info_ρR.converged == 0 || info_ΩL.converged == 0
        @warn "initial point not converged, ω = $Ω"
        @show info_ρR
        @show info_ΩL
    end
    x = (ΨL, ρR, ΩL, Ω, ω, ns, ωL)

    if verbosity > 0
        E = expval(density(Ĥ), ΨL, ρL, ρR)[]
        s = "UniformCMPS ground state: initalization with "
        s *= _groundstate_constraint_infostring(Ω, E, ns)
        @info s
    end

    x, Ω, normgrad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
    #                            precondition = precondition,
    #                            finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR) = x
    normgrad = sqrt(inner(x, grad, grad))
    e = expval(density(Ĥ), ΨL, ρL, ρR)
    E = e[]
    if verbosity > 0
        if normgrad <= gradtol
            s = @sprintf("UniformCMPS ground state: converged after %d iterations: ", size(history, 1))
        else
            s = "UniformCMPS ground state: not converged to requested tol: "
        end
        s *= _groundstate_constraint_infostring(Ω, E, ns, μs, normgrad)
        @info s
    end
    return ΨL, ρL, ρR, E, e, ns, μs, Ω, normgrad, numfg, history
end

function groundstate5(H::LocalHamiltonian, Ψ₀::UniformCMPS;
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

function groundstate6(H::LocalHamiltonian, Ψ₀::UniformCMPS, V, Ss;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, V, Ss, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dV, dSs = d

        Vold = V
        dX = dV * inv(V)
        #RLs = Ref(V) .* Ss .* Ref(inv(V))
        #@show dRs = Ref(dV) .* Ss .* Ref(inv(V)) .+ Ref(V) .* dSs .* Ref(inv(V)) .- Ref(V) .* Ss .* Ref(inv(V)) .* Ref(dV) .* Ref(inv(V))
        dRs = Ref(dX) .* RLs .+ Ref(V) .* dSs .* Ref(inv(V)) .- RLs .* Ref(dX)
        RdR = zero(KL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        V = Constant(exp(α * dX[])) * V
        Ss = Ss .+ α .* dSs

        RLs = Ref(V) .* Ss .* Ref(inv(V))
        KL = KL - (α/2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end
        dV = dV * inv(Vold) * V

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

        return (ΨL, V, Ss, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dV1, dSs1 = d1
        dV2, dSs2 = d2
        s = dV1 === dV2 ? 2*norm(dV1)^2 : 2*real(dot(dV1, dV2))
        for (dSs1,dSs2) in zip(dSs1, dSs2)
            if dSs1 === dSs2
                s += 2*norm(dSs1)^2
            else
                s += 2*real(dot(dSs1, dSs2))
            end
        end
        return s
    end

    function precondition(x, d)
        ΨL, V, Ss, ρR, = x
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
        (ΨL, V, Ss, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Q = ΨL.Q
        Rs = ΨL.Rs

        #dRs = gradRs
        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        dSs = Constant.(diagm.((diag.(broadcast(x->x[],(Ref(V') .* dRs .* Ref(inv(V)')))))))

        #dV = zero(V)
        #for (R, dR) in zip(Rs,dRs)
        #    dV += (dR*R' - R'*dR) * (inv(V))'
        #end

        dV = sum((dRs .* adjoint.(Rs) .- adjoint.(Rs) .* dRs) .* Ref(inv(V)'))

        return E, (dV, dSs)
    end

    function scale!(d, α)
        dV, dSs = d
        rmul!(dV, α)
        for dS in dSs
            rmul!(dS, α)
        end
        return d
    end
    function add!(d1, d2, α)
        dV1, dS1s = d1
        dV2, dS2s = d2
        axpy!(α, dV2, dV1)
        for (dS1, dS2) in zip(dS1s, dS2s)
            axpy!(α, dS2, dS1)
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
    #Rs = Ψ₀.Rs
    #V = Constant(eigvecs(Rs[1][]))
    #Ss = Constant.(diagm.(eigvals.(broadcast(x->x[],Rs))))

    x = (ΨL₀, V, Ss, ρR, HL, E, e, hL)

    #x, E, normgrad, numfg, history =
    #    optimize(fg, x, optalg; retract = retract,
    #                            precondition = precondition,
    #                            finalize! = _finalize!,
    #                            inner = inner, transport! = transport!,
    #                            scale! = scale!, add! = add!,
    #                            isometrictransport = true)
    #(ΨL, V, Ss, ρR, HL, E, e, hL) = x
    #return ΨL, ρR, E, e, normgrad, numfg, history
    return optimtest(fg, x; retract = retract, inner = inner)
end
