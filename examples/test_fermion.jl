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
multiple = false
interacting = true
Δ = 5.
B = 25.
test_2fermion = false

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
    QL = KL
    for R in RLs
        mul!(QL, R', R, -1/2, 1)
    end
    Ψ = InfiniteCMPS(QL, (R1,R2); gauge = :left)
    if interacting
        h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) - 2*Δ*(ψ[1]'*ψ[2]'*ψ[2]*ψ[1] + ψ[2]'*ψ[1]'*ψ[1]*ψ[2]) - B*(ψ[1]'*ψ[1] - ψ[2]'*ψ[2])
    else
        h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2])
    end
    H = ∫(h, (-Inf,+Inf))
else
    KL = Constant(randn(2*D,2*D))
    KL = 0.5*(KL-KL')
    R1 = Constant(kron(randn(D,D),σ⁺))
    RLs = (R1,)
    QL = KL
    for R in RLs
        mul!(QL, R', R, -1/2, 1)
    end
    Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    h = k * (∂ψ[1]'*∂ψ[1]) - μ * (ψ[1]'*ψ[1])
    H = ∫(h, (-Inf,+Inf))
end


alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-3);

if multiple
    Ψ, ρR, E, e, normgrad, numfg, history = groundstate_fermionbis(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    # αs,fs, dfs1, dfs2 = groundstate_fermionbis(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    # display(plot(αs,[dfs1,dfs2]))
    # gui()
    # μs = [5,4.5,4,3.5,3,2.5,2,1.5,1,0.5]*pi^2
    # Es = []
    # Ψs = []
    # histories = []
    #
    # let Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    #     for μi in μs
    #         μ = μi
    #         h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2])
    #         H = ∫(h, (-Inf,+Inf))
    #         #Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    #         Ψ, ρR, E, e, normgrad, numfg, history = groundstate_fermionbis(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #         push!(Es,E)
    #         push!(Ψs,Ψ)
    #         push!(histories, history)
    #     end
    # end
    #
    # @save "data2fermions_D=4_different_mus" μs Es Ψs histories
else
    Ψ, ρR, E, e, normgrad, numfg, history = groundstate_fermion(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    @show E, quadgk(f,-sqrt(μ),sqrt(μ))

    #αs,fs, dfs1, dfs2 = groundstate_fermion(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #display(plot(αs,[dfs1,dfs2]))
    #gui()
    # μs = [5,4.5,4,3.5,3,2.5,2,1.5,1,0.5]*pi^2
    # Es = []
    # Ψs = []
    # histories = []
    #
    # let Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    #     for μi in μs
    #         μ = μi
    #         h = k * (∂ψ[1]'*∂ψ[1]) - μ * (ψ[1]'*ψ[1])
    #         H = ∫(h, (-Inf,+Inf))
    #         #Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
    #         Ψ, ρR, E, e, normgrad, numfg, history = groundstate_fermion(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
    #         push!(Es,E)
    #         push!(Ψs,Ψ)
    #         push!(histories, history)
    #     end
    # end
    #
    #         @save "data1fermion_D=4_different_mus" μs Es Ψs histories
end

if test_2fermion
    Q = Ψ.Q
    R1 = Ψ.Rs[1]
    @show expval(ψ[1]*ψ[1],Ψ)[]
    @show expval(ψ[1]'*ψ[1],Ψ)[]
    @show E, quadgk(f,-sqrt(μ),sqrt(μ))

    σᶻ = [1. 0.; 0. -1.];
    Id = 1*Matrix(I,2,2);
    Id2 = 1*Matrix(I,D,D);
    q = Ψ.Q;
    r = Ψ.Rs[1];
    Q = Constant(kron(q[],Id2) + kron(Id2,q[]));
    R1 = Constant(kron(r[],Id2))
    R2 = Constant(kron(Id2,r[]))
    # Q = Constant(kron(q[],kron(Id2,Id)) + kron(kron(Id2,Id),q[]));
    # R1 = Constant(kron(r[],kron(Id2,Id)));
    # R2 = Constant(kron(kron(Id2,σᶻ),r[]));
    @show norm((R1*R1)[]),norm((R2*R2)[]),norm((R1*R2+R2*R1)[]),norm((R1*R2)[])
    @show norm(Q[]' + Q[] + (R1'*R1 + R2'*R2)[])
    Ψ = InfiniteCMPS(Q, (R1,R2); gauge = :left)
    h1 = k * (∂ψ[1]'*∂ψ[1]) - μ * (ψ[1]'*ψ[1])
    h2 = k * (∂ψ[2]'*∂ψ[2]) - μ * (ψ[2]'*ψ[2]);
    h = h1 + h2;
    @show expval(h1,Ψ)[]
    @show expval(h2,Ψ)[]
    @show expval(h,Ψ)[]
    @show expval(ψ[1]'*ψ[1],Ψ)[]
    @show expval(ψ[2]'*ψ[2],Ψ)[]
end

# Λ = 10000.
# KL = Constant(randn(2*D,2*D))
# KL = 0.5*(KL-KL')
# R1 = Constant(randn(2*D,2*D))
# R2 = Constant(randn(2*D,2*D))
# #Rls = (R1,)
# RLs = (R1,R2)
# QL = KL
# for R in RLs
#     mul!(QL, R', R, -1/2, 1)
# end
# Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
# #h = k * (∂ψ[1]'*∂ψ[1]) - μ * (ψ[1]'*ψ[1]) + Λ * (ψ[1]'*ψ[1]'*ψ[1]*ψ[1])
# h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + Λ * (2*ψ[1]'*ψ[1]'*ψ[1]*ψ[1] + 2*ψ[2]'*ψ[2]'*ψ[2]*ψ[2] + 4*(ψ[1]*ψ[2] + ψ[2]*ψ[1])'*(ψ[1]*ψ[2]+ψ[2]*ψ[1]))
# H = ∫(h, (-Inf,+Inf))
# Ψ, ρR, E, e, normgrad, numfg, history = groundstate(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
#
# @show Q = Ψ.Q
# @show R1 = Ψ.Rs[1]
# @show expval(ψ[1]*ψ[1],Ψ)[]
# @show E, quadgk(f,-sqrt(μ),sqrt(μ))
