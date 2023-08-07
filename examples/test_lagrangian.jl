#unique!(push!(LOAD_PATH, "~/Documents/UGent/PhD/Code_cMPS/julia1.0/CMPSKit.jl/src/"))
unique!(push!(LOAD_PATH, joinpath(pwd(), "src")))


using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2
using TensorOperations
using Plots

D = 8
global k = 1.
global μ = 2.3
global c = 1.5
#Λ = 10.

KL = Constant(randn(D,D))
KL = 0.5*(KL-KL')
R1 = Constant(randn(D,D))
R2 = Constant(randn(D,D))
RLs = (R1,R2)
QL = KL
for R in RLs
    mul!(QL, R', R, -1/2, 1)
end
#Put them in cMPS form
#Ψ = InfiniteCMPS(QL, RLs; gauge = :left)

#h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
#H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-4);
linalg = GMRES(krylovdim = 80; tol = 1e-5)

Λs = [1.,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
Es = []
Ψs = []
histories = []

let Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    for Λi in Λs
        Λ = Λi
        h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
        H = ∫(h, (-Inf,+Inf))
        #Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
        Ψ, ρR, E, e, normgrad, numfg, history = groundstate(H, Ψ; optalg = alg1, linalg = linalg)
        push!(Es,E)
        push!(Ψs,Ψ)
        push!(histories, history)
    end
end

@save "data_D=4" Λs Es Ψs histories
