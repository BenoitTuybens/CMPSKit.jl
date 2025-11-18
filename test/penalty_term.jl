using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2
using TensorOperations
using Plots
using LaTeXStrings

D = 16
k = 1.
μ1 = 2.0
μ2 = 2.0
c1 = 1.0
c2 = 1.0
c12 = 0.0
c21 = 0.0
Λ = 10000.

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
Ψ = InfiniteCMPS(QL, RLs; gauge = :left)


h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ1 * ψ[1]'*ψ[1] - μ2 * ψ[2]'*ψ[2] + c1 * (ψ[1]')^2*ψ[1]^2 + c2 * (ψ[2]')^2*ψ[2]^2 + c12 * ψ[1]'*ψ[2]'*ψ[2]*ψ[1] + c21 * ψ[2]'*ψ[1]'*ψ[1]*ψ[2] + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
H = ∫(h, (-Inf,+Inf))

alg1 = LBFGS(; verbosity = 4, maxiter = 20000, gradtol = 1e-3);
linalg = GMRES(krylovdim = 80; tol = 1e-5)

Λs = [1.,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
Es = []
Ψs = []
histories = []

# let Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
#     for Λi in Λs
#         Λ = Λi
#         h = k * (∂ψ[1]'*∂ψ[1] + ∂ψ[2]'*∂ψ[2]) - μ1 * ψ[1]'*ψ[1] - μ2 * ψ[2]'*ψ[2] + c1 * (ψ[1]')^2*ψ[1]^2 + c2 * (ψ[2]')^2*ψ[2]^2 + Λ * ((ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]))
#         H = ∫(h, (-Inf,+Inf))
#         # Ψ = InfiniteCMPS(QL, RLs; gauge = :left)
        Ψ, ρR, E, e, normgrad, numfg, history = groundstate(H, Ψ; optalg = alg1, linalg = linalg)
#         push!(Es,E)
#         push!(Ψs,Ψ)
#         push!(histories, history)
#     end
# end

# # @load "data_D=8" Λs Es
#
# @save "data_D=64" Λs Es Ψs histories
# @load "data_D=4" Λs Es Ψs histories
# @show length.(histories)
# E1s=Es
# @load "data_D=8" Λs Es Ψs histories
# @show length.(histories)
# E2s=Es
# @load "data_D=16" Λs Es Ψs histories
# @show length.(histories)
# E3s=Es
# @load "data_D=32" Λs Es Ψs histories
# @show length.(histories)
# E4s=Es
start = 100
@load "Data/mu1_mu2_2_c1_c2_1_c12_0/data_D=32" Λs Es Ψs histories
steps = range(start,size(histories[1])[1]-1, step=1)
plot(steps, histories[1][start:end-1,1], label="Λ=1e1", ylabel="E", xlabel="steps")
steps = range(start,size(histories[2])[1]-1, step=1)
plot!(steps, histories[2][start:end-1,1], label="Λ=1e2", ylabel="E", xlabel="steps")
steps = range(start,size(histories[3])[1]-1, step=1)
plot!(steps, histories[3][start:end-1,1], label="Λ=1e3", ylabel="E", xlabel="steps")
steps = range(start,size(histories[4])[1]-1, step=1)
plot!(steps, histories[4][start:end-1,1], label="Λ=1e4", ylabel="E", xlabel="steps")
steps = range(start,size(histories[5])[1]-1, step=1)
plot!(steps, histories[5][start:end-1,1], label="Λ=1e5", ylabel="E", xlabel="steps")
steps = range(start,size(histories[6])[1]-1, step=1)
plot!(steps, histories[6][start:end-1,1], label="Λ=1e6", ylabel="E", xlabel="steps")
steps = range(start,size(histories[7])[1]-1, step=1)
plot!(steps, histories[7][start:end-1,1], label="Λ=1e7", ylabel="E", xlabel="steps")
steps = range(start,size(histories[8])[1]-1, step=1)
display(plot!(steps, histories[8][start:end-1,1], label="Λ=1e8", ylabel="E", xlabel="steps"))
# gui()
# E5s=Es
# plot!(Λs,E1s,label="D=4", xaxis=:log, seriestype=:scatter)
# plot!(Λs,E2s,label="D=8", xaxis=:log, seriestype=:scatter)
# plot!(Λs,E3s,label="D=16", xaxis=:log, seriestype=:scatter)
# plot!(Λs,E4s,label="D=32", xaxis=:log, seriestype=:scatter)
# xlabel!(L"\Lambda")
title!("Optimalisation using Lagrangian multiplier D = 32")
plot!(legend=:bottomright)
# display(plot!(Λs,E3s, label="D=16", ylabel="E", xaxis=:log, seriestype=:scatter,yguidefontrotation=-90,margin = 2Plots.mm))
savefig("lagrangianhistory.pdf")
# gui()