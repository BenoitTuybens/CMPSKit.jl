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

leftgauge = false
if leftgauge
    #Tensors product ansatz
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
else
    #Tensors product ansatz
    Id = 1*Matrix(I,D,D)
    Q = Constant(randn(D^2,D^2))
    R1 = Constant(kron(Id,randn(D,D)))
    R2 = Constant(kron(randn(D,D),Id))
    #Put them in cMPS form
    Ψ = InfiniteCMPS(Q, (R1,R2))
end

alg1 = LBFGS(; verbosity = 20, maxiter = 1000000, gradtol = 1e-3);
linalg = GMRES(krylovdim = 100; tol = 1e-5)
h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + c * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2)
H = ∫(h, (-Inf,+Inf))

if leftgauge
    Ψ, ρR, E, e, normgrad, numfg, history = groundstate4_unconstrained(H, Ψ; optalg = alg1, linalg = linalg)
    #αs,fs, dfs1, dfs2 = groundstate4(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #αs = (αs[1:end-1] + αs[2:end])/2
    #push!(αs,0.1)
    #display(plot(αs,[dfs1,dfs2]))
    #gui()
else
    Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate3(H, Ψ; optalg = alg1, linalg = linalg)
    #αs,fs, dfs1, dfs2 = groundstate3(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #αs = (αs[1:end-1] + αs[2:end])/2
    #push!(αs,0.1)
end

@show Q = Ψ.Q
@show R1 = Ψ.Rs[1]
@show R2 = Ψ.Rs[2]
@show R1[]*R2[] - R2[]*R1[]
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
