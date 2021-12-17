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

x = randn(D,D)
f = qr(x)
diagR = sign.(real(diag(f.R)))
diagR[diagR.==0] .= 1
diagRm = diagm(diagR)
V = f.Q * diagRm

V = Constant(V)
S1 = Constant(diagm(rand(D)/D))
S2 = Constant(diagm(rand(D)/D))
Q = Constant(Matrix(Symmetric(randn(D,D))))
#Q = Constant(randn(D,D))
R1 = V*S1*inv(V)
R2 = V*S2*inv(V)
Ss = (S1,S2)
Rs = (R1,R2)

#Put them in cMPS form
Ψ = InfiniteCMPS(Q, Rs)

h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2)
H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-3);
linalg = GMRES(krylovdim = 80; tol = 1e-5)

Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate7(H, Ψ, V, Ss; optalg = alg1, linalg = linalg)
# αs,fs, dfs1, dfs2 = groundstate7(H, Ψ, V, Ss; optalg = alg1, linalg = linalg)
# display(plot(αs,[dfs1,dfs2]))
# gui()
