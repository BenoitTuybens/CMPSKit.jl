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
using QuadGK

function f(x)
    return (x^2-μ)/(2*pi)
end

D = 4
k = 1.
μ = 5.

σ⁺ = [0. 1.; 0. 0.]
σˣ = [0. 1.; 1. 0.]
Id = 1*Matrix(I,2,2)

Q = Constant(kron(Id,randn(D,D)))
R1 = Constant(kron(σˣ,randn(D,D)))
Rs = (R1,)

Ψ = InfiniteCMPS(Q, Rs)
h = k * (∂ψ[1]'*∂ψ[1]) - μ * (ψ[1]'*ψ[1])
H = ∫(h, (-Inf,+Inf))


alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-3);

Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate_fermion_sigmaplus(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
#αs,fs, dfs1, dfs2 = groundstate_fermion_sigmaplus(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
#display(plot(αs,[dfs1,dfs2]))
#gui()
@show E, quadgk(f,-sqrt(μ),sqrt(μ))
