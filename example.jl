include("BayesCMPM.jl")

m = ModelCMPM()

b = Bayesian()

msim = @vam(time & type ~ (ARA∞(0.2) | Weibull(1.0, 2.0)) &  (ARA∞(0.7) | Periodic(2)) )

df = simulate(msim, @stop(size < 100))

VirtualAgeModels.params(msim)
mcmc(b, VirtualAgeModels.params(msim), data=df, burn=0)
b.estimates
mean(b.estimates[b.estimates.ind .== 4,2])
b.estimates[b.estimates.ind .== 2,2]
mle(msim,df)

using Plots

plot(b.estimates[b.estimates.ind .== 2,2])