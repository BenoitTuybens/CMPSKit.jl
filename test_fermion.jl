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
μ = (pi)^2
multiple = false

σ⁺ = [0. 1.; 0. 0.]

if multiple
    σᶻ = [1. 0.; 0. -1.]
    Id = 1*Matrix(I,2,2)
    Id2 = 1*Matrix(I,D,D)
    KL = Constant(randn(4*D^2,4*D^2))
    KL = 0.5*(KL-KL')
    R1 = Constant(kron(kron(kron(randn(D,D),Id2),σ⁺),Id))
    R2 = Constant(kron(kron(kron(Id2,randn(D,D)),σᶻ),σ⁺))
    RLs = (R1,R2)
    Q = Constant(randn(2*D^2,2*D^2))
    QL = KL
    for R in RLs
        mul!(QL, R', R, -1/2, 1)
    end
    Ψ = InfiniteCMPS(QL, (R1,R2); gauge = :left)
    h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2])
else
    KL = Constant(randn(2*D,2*D))
    KL = 0.5*(KL-KL')
    R1 = Constant(kron(randn(D,D),σ⁺))
    RLs = (R1,)
    QL = KL
    for R in RLs
        mul!(QL, R', R, -1/2, 1)
    end
    Ψ = InfiniteCMPS(QL, (R1,); gauge = :left)
    h = k * (∂ψ[1]'*∂ψ[1]) - μ * (ψ[1]'*ψ[1])
end

H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-4);

if multiple
    #Ψ, ρR, E, e, normgrad, numfg, history = groundstate5bis(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    αs,fs, dfs1, dfs2 = groundstate5bis(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    display(plot(αs,[dfs1,dfs2]))
    gui()
else
    Ψ, ρR, E, e, normgrad, numfg, history = groundstate5(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #αs,fs, dfs1, dfs2 = groundstate5(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #display(plot(αs,[dfs1,dfs2]))
    #gui()
end

@show Q = Ψ.Q
@show R1 = Ψ.Rs[1]
@show expval(ψ[1]*ψ[1],Ψ)[]
