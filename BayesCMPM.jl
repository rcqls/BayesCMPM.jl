using VirtualAgeModels
using DataFrames
using Distributions

mutable struct ModelCMPM
    ARA1CM::Bool 
    ARA1PM::Bool

    aα::Float64 # shape Gamma
    bα::Float64 # rate Gamma
    aT1::Float64 # shape InvGamma
    bT1::Float64 # scale InvGamma

    aβ::Float64 # lower bound Unif
    bβ::Float64 # upper bound Unif
    aCM::Float64 # Beta
    bCM::Float64 # Beta
    aPM::Float64 # Beta
    bPM::Float64 # Beta
    δ::Bool # true => ρCM < ρPM
    γ::Bool # true => Non Informative
end

ModelCMPM(;ARA1CM=false,ARA1PM=false,aα=3.0,bα=0.5,aT1=-1.0,bT1=-1.0,aβ=1.0,bβ=5.0,aCM=2.5,bCM=5.83,aPM=4.89,bPM=1.22,δ=false,γ=false) = ModelCMPM(ARA1CM,ARA1PM,aα,bα,aT1,bT1,aβ,bβ,aCM,bCM,aPM,bPM,δ,γ) 

isinformative(m::ModelCMPM) = !m.γ

mutable struct Bayesian
    model::ModelCMPM
    estimates::DataFrame
    mle::VirtualAgeModels.MLE
    σ::Float64
    θ::Vector{Float64} # θ = [α, β, ρCM, ρPM]
    profile::Bool
end

function Bayesian(;model=ModelCMPM(),estimates=DataFrame(),mle=VirtualAgeModels.MLE(),σ=0.1,profile=false)
    b = Bayesian(model,estimates,mle,σ,Float64[],profile)
    m = if b.model.ARA1CM && b.model.ARA1PM
        @vam(time & type ~ (ARA1(0.5) | Weibull(1.0, 2.0)) &  ARA1(0.5))
    elseif b.model.ARA1CM && !b.model.ARA1PM
        @vam(time & type ~ (ARA1(0.5) | Weibull(1.0, 2.0)) &  ARA∞(0.5))
    elseif !b.model.ARA1CM && b.model.ARA1PM
        @vam(time & type ~ (ARA∞(0.5) | Weibull(1.0, 2.0)) &  ARA1(0.5))
    else
        @vam(time & type ~ (ARA∞(0.5) | Weibull(1.0, 2.0)) &  ARA∞(0.5))
    end
    b.mle = VirtualAgeModels.MLE(m)
    return b
end

import Base.rand
rand(b::Bayesian, μ::Float64) = rand(Normal(μ,b.σ))

abstract type Postα end
abstract type Postβ end
abstract type PostρCM end
abstract type PostρPM end

const PostTypes = [Postα, Postβ, PostρCM, PostρPM]

function compute(b::Bayesian, θ::Vector{Float64}) # TODO profile????
    contrast(b.mle, θ)
end

function logpdf(b::Bayesian,β::Float64,::Type{Postβ}; mean=false)
    # println(b.θ)
   return  (b.model.aβ < β < b.model.bβ) ?  (-b.θ[1] * b.mle.comp.S1 + (b.mle.comp.S0 - b.model.γ) * log(β) + (β - 1) * (b.mle.comp.S2 + b.mle.comp.S3)) : -Inf
end

function logpdf(b::Bayesian,ρ::Float64,::Type{PostρCM}; mean=false)
    return ( 0.0 < ρ < 1.0) ?  -b.θ[1] + (b.θ[2] - 1) * (b.mle.comp.S2 + b.mle.comp.S3) + (b.model.aCM - 1) * log(ρ) + (b.model.bCM - 1) * log(1 - ρ) : -Inf
end

function logpdf(b::Bayesian,ρ::Float64,::Type{PostρPM}; mean=false)
    return ( b.model.δ * b.θ[3] < ρ < 1.0) ?  -b.θ[1] + (b.θ[2] - 1) * (b.mle.comp.S2 + b.mle.comp.S3) + (b.model.aPM - 1) * log(ρ) + (b.model.bPM - 1) * log(1 - ρ) : -Inf
end

function mcmc(b::Bayesian, θ::Vector{Float64}; data::DataFrame=DataFrame(), nb::Int=10000, burn::Int=1000)
    profile = b.profile
    data!(b.mle.model, data)
    b.θ = θ
    nbparams = length(θ)
    # priors_ = priors(b.model)
    curθ, oldθ =copy(θ), copy(θ)
    ind, θhat, αhat = Int[], Float64[], Float64[]
    for i in 1:nb
        save = false
        for j in 1+profile:nbparams
            compute(b,curθ)
            if j==1 && !profile
                # TODO
                curθ[j]=rand(Gamma(b.model.aα + b.mle.comp.S0, 1.0 / (b.model.bα + b.mle.comp.S1)))
                save = true
                # compute(b,curθ)
            else
                curθ[j]=rand(b,oldθ[j])
                # compute(b,curθ)
                logr = logpdf(b, curθ[j], PostTypes[j]) - logpdf(b, oldθ[j], PostTypes[j])
                if logr > log(rand())
                    save = true
                    oldθ[j] = curθ[j]
                else
                    curθ[j] = oldθ[j]
                end
            end
            if i >= burn && save
                push!(ind, j)
                push!(θhat,curθ[j])
                if profile
                    push!(αhat, VirtualAgeModels.αEst(b.mle,curθ))
                end
            end
        end
    end
    res = DataFrame(ind=ind, θ=θhat)
    if profile
        res[!,:α] = αhat
    end
    b.estimates = res
    return res
end
