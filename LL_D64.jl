#unique!(push!(LOAD_PATH, "~/Documents/UGent/PhD/Code_cMPS/julia1.0/CMPSKit.jl/src/"))
unique!(push!(LOAD_PATH, joinpath(pwd(), "src")))


using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2

D = 64

#Q = Constant(randn(D,D))
#R = Constant(randn(D,D))
#Ψ = InfiniteCMPS(Q, R)

@load "DataQRD64_final" ΨL ρR HL E e hL


#alg = LBFGS(; verbosity = 2, maxiter = 10^6, gradtol = 1e-4);

Q = ΨL.Q
R = (ΨL.Rs)[1]

H = ∫(∂ψ[1]'*∂ψ[1] - 10000 * (ψ[1]'*ψ[1]) + 1000 * (ψ[1]'*ψ[1]'*ψ[1]*ψ[1]), (-Inf,+Inf))

ρL = one(ρR)
Z = tr(ρR)[]
_, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR))
density = tr(R*ρR*R')[]

function environment(ΨL,ρL,ρR)
    linalg = GMRES(; tol = 1e-4)
    R = (ΨL.Rs)[1]
    density = tr(R*ρR*R')[]
    hL = R'*ρL*R
    hL = axpy!(-density, ρL, hL)

    HL₀ = zero(Q)

    let TL = LeftTransfer(ΨL)
        tol = linalg.tol
        HL, infoL = linsolve(hL, HL₀, linalg) do x
            y = ∂(x) - TL(x; tol = tol/10)
            y = axpy!(dot(ρR, x), ρL, y)
            CMPSKit.truncate!(y; tol = tol/10)
        end

        HL = rmul!(HL + HL', 0.5)
        HL = CMPSKit.truncate!(HL; tol = tol/10)
        return HL
    end
end
Env = environment(ΨL,ρL,ρR)
R² = dot(R,R)
@show Z
@show E
@show density
@show density² = density^2
@show tr(Env*ρR)
@show 2*tr(Env*R*ρR*R')[]
@show tr(R²*ρR*R²')
@show variance = sqrt(2*tr(Env*R*ρR*R')[] + density)
n̂ = ψ'*ψ
N̂ = ∫(n̂, (-Inf,+Inf));
@show N = expval(n̂, ΨL, ρL, ρR)
N = tr(R*ρR*R')[]
@show sqrt(2*tr(leftenv(N̂, (ΨL,ρL,ρR))[1]*R*ρR*R')[] + N)

@show psi = expval(ψ, ΨL, ρL, ρR)

# groundstate(H, Ψ; optalg = alg, linalg = GMRES(; tol = 1e-4), finalize! = finalize!)
#ΨL, ρR, E, e, normgrad, numfg, history = groundstate(H, Ψ; optalg = alg, linalg = GMRES(; tol = 1e-6))
