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


V = Constant(randn(D,D))
S1 = Constant(diagm(rand(D)/D))
S2 = Constant(diagm(rand(D)/D))
KL = Constant(randn(D,D))
KL = 0.5*(KL-KL')
R1 = V*S1*inv(V)
R2 = V*S2*inv(V)
Ss = (S1,S2)
RLs = (R1,R2)
QL = KL
for R in RLs
    mul!(QL, R', R, -1/2, 1)
end
#Put them in cMPS form
Ψ = InfiniteCMPS(QL, (R1,R2); gauge = :left)

h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2)
H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-4);
linalg = GMRES(krylovdim = 80; tol = 1e-5)

#Ψ, ρR, E, e, normgrad, numfg, history = groundstate6(H, Ψ, V, Ss; optalg = alg1, linalg = linalg)
αs,fs, dfs1, dfs2 = groundstate6(H, Ψ, V, Ss; optalg = alg1, linalg = linalg)
#αs = (αs[1:end-1] + αs[2:end])/2
#push!(αs,0.1)
display(plot(αs,[dfs1,dfs2]))
gui()
