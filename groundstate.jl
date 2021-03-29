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
μ = 1.
g = 1.
Λ = 1000.
gradienttest = false
tensorproduct = true
optimisation = true
lagrangianmultiplier = false

if tensorproduct
    #Tensors product ansatz
    Id = 1*Matrix(I,D,D)
    Q = Constant(randn(D^2,D^2))
    R1 = Constant(kron(Id,randn(D,D)))
    R2 = Constant(kron(randn(D,D),Id))
else
    #Random Tensors
    Q = Constant(randn(D^2,D^2))
    R1 = Constant(randn(D^2,D^2))
    R2 = Constant(randn(D^2,D^2))
end

function firstordercorrection(dQ,gradQ,dR1,gradR1,dR2,gradR2)
    #left gauge: dQ = -R1'*dR1 - R2'*dR2
    return 2*real(dot(dQ,gradQ) + dot(dR1,gradR1) + dot(dR2,gradR2));
end

#Put them in cMPS form
Ψ = InfiniteCMPS(Q, (R1,R2))
alg1 = LBFGS(; verbosity = 2, maxiter = 1000000, gradtol = 1e-4);
alg2 = ConjugateGradient(; verbosity = 2, maxiter = 1000000, gradtol = 1e-4);
#Lagrangian multiplier

if lagrangianmultiplier
    h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
    H = ∫(k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1])), (-Inf,+Inf))
else
    h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2)
    H = ∫(k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2), (-Inf,+Inf))
end

#h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
#H = ∫(k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2) + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1])), (-Inf,+Inf))
#h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2)
#H = ∫(k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ * (ψ[1]'*ψ[1] + ψ[2]'*ψ[2]) + g * ((ψ[1]')^2*ψ[1]^2 + (ψ[2]')^2*ψ[2]^2), (-Inf,+Inf))

if gradienttest
    ρL, ρR, infoL, infoR = environments!(Ψ;linalg = GMRES(; tol = 1e-6))

    HL, E, e, hL, infoL = leftenv(H, (Ψ,ρL,ρR);linalg = GMRES(; tol = 1e-6))
    HR, E, e, hR, infoR = rightenv(H, (Ψ,ρL,ρR);linalg = GMRES(; tol = 1e-6))

    gradQ, gradRs = gradient_tens_prod(H, (Ψ, ρL, ρR))
    dQ = Constant(1e-4*randn(D^2,D^2))
    dR1 = Constant(kron(Id,1e-4*(randn(D,D))))
    dR2 = Constant(kron(1e-4*(randn(D,D)),Id))
    #left gauge dK = -1im*(dR1'*R1 - R1'*dR1 + dR2'*R2 - R2'*dR2)/2
    Ψ2 = InfiniteCMPS(Q+dQ,(R1+dR1,R2+dR2))
    foc1 = firstordercorrection(dQ,gradQ,dR1,gradRs[1],dR2,gradRs[2])
    dQ/=10
    dR1/=10
    dR2/=10
    foc2 = firstordercorrection(dQ,gradQ,dR1,gradRs[1],dR2,gradRs[2])
    Ψ3 = InfiniteCMPS(Q+dQ,(R1+dR1,R2+dR2))
    dQ/=10
    dR1/=10
    dR2/=10
    foc3 = firstordercorrection(dQ,gradQ,dR1,gradRs[1],dR2,gradRs[2])
    Ψ4 = InfiniteCMPS(Q+dQ,(R1+dR1,R2+dR2))

    @show E1 = expval(h,Ψ)
    E2 = expval(h,Ψ2)
    E3 = expval(h,Ψ3)
    E4 = expval(h,Ψ4)

    @show E2[]-E1[]-foc1
    @show E3[]-E1[]-foc2
    @show E4[]-E1[]-foc3
end


if optimisation

    if tensorproduct
        Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate3(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
        #αs,fs, dfs1, dfs2 = groundstate3(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
        #αs = (αs[1:end-1] + αs[2:end])/2
        #push!(αs,0.1)
    else
        Ψ, ρR, E, e, normgrad, numfg, history = groundstate(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
        #αs,fs, dfs1, dfs2 = groundstate(H, Ψ; optalg = alg1, linalg = GMRES(; tol = 1e-5))
        #αs = (αs[1:end-1] + αs[2:end])/2
        #push!(αs,0.1)
    end

    #display(plot(αs,[dfs1,dfs2]))
    #gui()
    @show Q = Ψ.Q
    @show R1 = Ψ.Rs[1]
    @show R2 = Ψ.Rs[2]
    @show R1[]*R2[] - R2[]*R1[]
    @show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
    #DR1 = differentiate(R1) + Q * R1 - R1 * Q
    #DR2 = differentiate(R2) + Q * R2 - R2 * Q
    #R1² = R1 * R1
    #R2² = R2 * R2
end
