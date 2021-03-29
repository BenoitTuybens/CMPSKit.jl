function gradient(H::LocalHamiltonian, Ψρs::InfiniteCMPSData, HL = nothing, HR = nothing;
                    kwargs...)
    Ψ, ρL, ρR = Ψρs
    if isnothing(HL)
        HL, = leftenv(H, Ψρs; kwargs...)
    end
    if isnothing(HR)
        HR, = rightenv(H, Ψρs; kwargs...)
    end

    Q = Ψ.Q
    Rs = Ψ.Rs

    gradQ = zero(Q)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        gradQ = gradQ + coeff * localgradientQ(op, Q, Rs, ρL, ρR)
    end
    gradQ += HL*ρR + ρL*HR

    gradRs = zero.(Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        gradRs = gradRs .+ Ref(coeff) .* localgradientRs(op, Q, Rs, ρL, ρR)
    end
    gradRs = gradRs .+ Ref(HL) .* Rs .* Ref(ρR) .+ Ref(ρL) .* Rs .* Ref(HR)

    return gradQ, gradRs
end

function gradient_tens_prod(H::LocalHamiltonian, Ψρs::InfiniteCMPSData, HL = nothing, HR = nothing;
                    kwargs...)

    Ψ, ρL, ρR = Ψρs
    if isnothing(HL)
        HL, = leftenv(H, Ψρs; kwargs...)
    end
    if isnothing(HR)
        HR, = rightenv(H, Ψρs; kwargs...)
    end

    Q = Ψ.Q
    D = Int(sqrt(size(Q[])[1]))
    Rs = Ψ.Rs

    gradQ = zero(Q)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        gradQ = gradQ + coeff * localgradientQ(op, Q, Rs, ρL, ρR)
    end
    gradQ += HL*ρR + ρL*HR

    gradRs = zero.(Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        gradRs = gradRs .+ Ref(coeff) .* localgradientRs(op, Q, Rs, ρL, ρR)
    end
    gradRs = gradRs .+ Ref(HL) .* Rs .* Ref(ρR) .+ Ref(ρL) .* Rs .* Ref(HR)

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

    return gradQ, gradRs
end
