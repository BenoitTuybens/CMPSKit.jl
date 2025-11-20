using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2
using TensorOperations
using Plots
using Random

D = 4
k = 1.
μ = 2.0
c = 1.0
c12 = 0.0

V = Constant(rand(ComplexF64,D,D))
S1 = Constant(diagm(rand(ComplexF64,D)/D))
S2 = Constant(diagm(rand(ComplexF64,D)/D))
KL = Constant(rand(ComplexF64,D,D))

KL = 0.5*(KL-KL')
R1 = V*S1*inv(V)
R2 = V*S2*inv(V)
Ss = (S1,S2)
RLs = (R1,R2)
QL = KL - 0.5*R1'*R1 - 0.5*R2'*R2

Ψ = InfiniteCMPS(QL, (R1,R2); gauge = :left)

h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + c12 * ((ψ[1]'*ψ[2]'*ψ[2]*ψ[1] + ψ[2]'*ψ[1]'*ψ[1]*ψ[2]))
# h2 = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + c12 * ((ψ[1]'*ψ[2]'*ψ[2]*ψ[1] + ψ[2]'*ψ[1]'*ψ[1]*ψ[2])) + 100000 * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
H = ∫(h, (-Inf,+Inf))
# H2 = ∫(h2, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 4, maxiter = 100000, gradtol = 1e-2);
alg2 = LBFGS(; verbosity = 4, maxiter = 100000, gradtol = 1e-3);
alg3 = LBFGS(; verbosity = 4, maxiter = 100000, gradtol = 1e-4);
linalg = GMRES(krylovdim = 80; maxiter = 5000, tol = 1e-5);

function expand(ψ::InfiniteCMPS; ϵ = 0.05)
    D = size(Ψ.Rs[1][],1)
    D_new = 2 * D

    I2 = Diagonal(ones(Float64, 2))
    V = eigvecs(Ψ.Rs[1][])
    S1 = diagm(diag((inv(V) * Ψ.Rs[1][] * V)))
    S2 = diagm(diag((inv(V) * Ψ.Rs[2][] * V)))
    Ss = (S1,S2)

    Ss_new = map(1:2) do ix
        kron(I2,Ss[ix]) 
    end
    Q_new = kron(I2, ψ.Q[]) 
    V_new = kron(I2, V)
    Rs_new = [V_new * S * inv(V_new) for S in Ss_new]

    pert = rand(ComplexF64, D_new, D_new)
    pert = (pert + pert') / norm(pert + pert')

    V_new = V_new * exp(ϵ * pert)
    Ss_new = map(Ss_new) do D
        dD = Diagonal(rand(ComplexF64, D_new))
        dD = (dD + dD') / norm(dD + dD')
        D + ϵ * dD
    end

    ΔRs = [V_new * S * inv(V_new) - R0 for (S, R0) in zip(Ss_new, Rs_new)]
    M = sum(-[R0' * ΔR + 0.5 * ΔR' * ΔR for (R0, ΔR) in zip(Rs_new, ΔRs)])
    Q_new = Q_new + M

    return InfiniteCMPS(Constant(Q_new), (Constant(Rs_new[1]), Constant(Rs_new[2])); gauge = :left), Constant(V_new), (Constant(Ss_new[1]),Constant(Ss_new[2]))
end

Ψ, ρR, E1, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ, V, Ss; optalg = alg1, linalg = linalg)
@show E1
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]

Ψ_new, V_new, Ss_new = expand(Ψ)
Ψ, ρL, ρR, E2, e, normgrad, numfg, history = groundstate_diagonal2(H, Ψ_new; optalg = LBFGS(; verbosity = 4, maxiter = 500, gradtol = 1e-3), linalg = linalg)

V = eigvecs(Ψ.Rs[1][])
S1 = diagm(diag((inv(V) * Ψ.Rs[1][] * V)))
S2 = diagm(diag((inv(V) * Ψ.Rs[2][] * V)))
Ss = (S1,S2)
Ψ_new, V_new, Ss_new = Ψ, V, Ss
Ψ, ρR, E2, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ_new, V_new, Ss_new; optalg = alg2, linalg = linalg)
@show E2
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]

Ψ_new, V_new, Ss_new = expand(Ψ)
Ψ, ρR, E, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ_new, V_new, Ss_new; optalg = alg3, linalg = linalg)

@show E
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]
