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

D = 4
k = 1.
μ = 2.3
c = 1.5
g = 2.

Id = 1*Matrix(I,D,D)
KL = Constant(randn(D^2,D^2))
KL = 0.5*(KL-KL')
R1 = Constant(kron(Id,randn(D,D)))
R2 = Constant(kron(randn(D,D),Id))
RLs = (R1,R2)
QL = KL
for R in RLs
    mul!(QL, R', R, -1/2, 1)
end
#Put them in cMPS form
Ψ = InfiniteCMPS(QL, (R1,R2); gauge = :left)

#h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + g * (ψ[1]'*ψ[2]'*ψ[1]*ψ[2])
#H = ∫(h, (-Inf,+Inf))

h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2)
H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-4);

#Ψ, ρR, E, e, normgrad, numfg, history = groundstate_tensprod_left(H, Ψ, (0.63,0.63); optalg = alg1, linalg = linalg)
αs,fs, dfs1, dfs2 = groundstate_tensprod_left(H, Ψ, (0.63,0.63); optalg = alg1, linalg = GMRES(; tol = 1e-5))
αs = (αs[1:end-1] + αs[2:end])/2
push!(αs,0.1)
display(plot(αs,[dfs1,dfs2]))
gui()
