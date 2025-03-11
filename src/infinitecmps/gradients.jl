function gradient(H::LocalHamiltonian, Ψρs::InfiniteCMPSData, HL=nothing, HR=nothing;
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

    # gradQ = ∑(coeff * localgradientQ)  +  HL*ρR + ρL*HR
    gradQ = zero(Q)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!(coeff, localgradientQ(op, Ψ, ρL, ρR), gradQ)
        else
            mul!(gradQ, coeff, localgradientQ(op, Ψ, ρL, ρR), 1, 1)
        end
    end
    mul!(gradQ, HL, ρR, 1, 1)
    mul!(gradQ, ρL, HR, 1, 1)

    # gradR = ∑(coeff * localgradientR)  +  HL*R*ρR + ρL*R*HR
    gradRs = zero.(Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!.(coeff, localgradientRs(op, Ψ, ρL, ρR), gradRs)
            if op isa ContainsDifferentiatedCreation && !(Q isa Constant)
                grad∂Rs = localgradient∂Rs(op, Ψ, ρL, ρR)
                axpy!.(-coeff, ∂.(grad∂Rs), gradRs)
            end
        else
            mul!.(gradRs, (coeff,), localgradientRs(op, Ψ, ρL, ρR), 1, 1)
            if op isa ContainsDifferentiatedCreation && !(Q isa Constant)
                grad∂Rs = localgradient∂Rs(op, Ψ, ρL, ρR)
                mul!.(gradRs, (-coeff,), ∂.(grad∂Rs), 1, 1)
            end
        end
    end
    mul!.(gradRs, (HL,), Rs .* (ρR,), 1, 1)
    mul!.(gradRs, (ρL,), Rs .* (HR,), 1, 1)

    return gradQ, gradRs
end

function centergradient(H::LocalHamiltonian, ΨLRC, HL=nothing, HR=nothing;
    kwargs...)
    ΨL, ΨR, C = ΨLRC
    if isnothing(HL)
        HL, = leftenv(H, (ΨL, one(C), C * C'); kwargs...)
    end
    if isnothing(HR)
        HR, = rightenv(H, (ΨR, C' * C, one(C)); kwargs...)
    end

    QL, RLs = ΨL
    QC = QL * C
    RCs = RLs .* (C,)

    # gradC = ∑(coeff * localgradientQC)  +  HL*C + C*HR
    gradC = zero(QC)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!(coeff, localgradientQC(op, ΨL, ΨR, C), gradC)
        else
            mul!(gradC, coeff, localgradientQC(op, ΨL, ΨR, C), 1, 1)
        end
    end
    mul!(gradC, HL, C, 1, 1)
    mul!(gradC, C, HR, 1, 1)

    # gradR = ∑(coeff * localgradientR)  +  HL*R*ρR + ρL*R*HR
    gradRCs = zero.(RCs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!.(coeff, localgradientRCs(op, ΨL, ΨR, C), gradRCs)
            # TODO: this doesn't probably make sense or is incorrect in a nonuniform case
            if op isa ContainsDifferentiatedCreation && !(QL isa Constant)
                grad∂RCs = localgradient∂RCs(op, ΨL, ΨR, C)
                axpy!.(-coeff, ∂.(grad∂RCs), gradRCs)
            end
        else
            mul!.(gradRCs, (coeff,), localgradientRCs(op, ΨL, ΨR, C), 1, 1)
            # TODO: this doesn't probably make sense or is incorrect in a nonuniform case
            if op isa ContainsDifferentiatedCreation && !(QL isa Constant)
                grad∂RCs = localgradient∂RCs(op, ΨL, ΨR, C)
                mul!.(gradRCs, (-coeff,), ∂.(grad∂RCs), 1, 1)
            end
        end
    end
    mul!.(gradRCs, (HL,), RCs, 1, 1)
    mul!.(gradRCs, RCs, (HR,), 1, 1)

    gradQC = -sum(adjoint.(RLs) .* gradRCs) - QL' * gradC # == - sum(gradRCs .* adjoint.(RRs)) - gradC * QR'
    return gradC, gradQC, gradRCs
end
