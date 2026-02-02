using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2
using TensorOperations
using Plots
using LaTeXStrings

χ = 32
k = 1.
μ1 = 2.0
μ2 = 2.0
c1 = 1.0
c2 = 1.0
c12 = 0.0
c21 = 0.0
Λ = 1000.0

KL = Constant(randn(χ,χ))
KL = 0.5*(KL-KL')
R1 = Constant(randn(χ,χ))
R2 = Constant(randn(χ,χ))
RLs = (R1,R2)
QL = KL
for R in RLs
    mul!(QL, R', R, -1/2, 1)
end
#Put them in cMPS form
Ψ = InfiniteCMPS(QL, RLs; gauge = :left)


h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ1 * ψ[1]'*ψ[1] - μ2 * ψ[2]'*ψ[2] + c1 * (ψ[1]')^2*ψ[1]^2 + c2 * (ψ[2]')^2*ψ[2]^2 + c12 * ψ[1]'*ψ[2]'*ψ[2]*ψ[1] + c21 * ψ[2]'*ψ[1]'*ψ[1]*ψ[2] + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 4, maxiter = 20000, gradtol = 1e-3);
linalg = GMRES(krylovdim = 80; tol = 1e-5)

Λs = [1.,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
Es = []
Ψs = []
histories = []

let Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    for Λi in Λs
        Λ = Λi
        h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ1 * ψ[1]'*ψ[1] - μ2 * ψ[2]'*ψ[2] + c1 * (ψ[1]')^2*ψ[1]^2 + c2 * (ψ[2]')^2*ψ[2]^2 + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
        H = ∫(h, (-Inf,+Inf))
        # Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
        Ψ, ρR, E, e, normgrad, numfg, history = groundstate(H, Ψ; optalg = alg1, linalg = linalg)
        push!(Es,E)
        push!(Ψs,Ψ)
        push!(histories, history)
    end
end

@save "data_D=$(χ)" Λs Es Ψs histories