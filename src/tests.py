
pyro.clear_param_store()
def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(.1)
    beta0 = torch.tensor(100.)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    with pyro.plate("data", data.shape[0]):
        pyro.sample("obs", dist.Bernoulli(f), obs=data)

guide = AutoDiagonalNormal(model)

#def guide(data):
#    alpha_q = pyro.param("alpha_q", torch.tensor(15.0), constraint=constraints.positive)
#    beta_q = pyro.param("beta_q", torch.tensor(15.0), constraint=constraints.positive) 
#    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

data = torch.tensor( [ 1.0 for _ in range(500) ] + [ 0.0 for _ in range(4) ] )

optimizer = pyro.optim.Adam({})

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(5000):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')
        
#guide.get_posterior()
#guide.get_transform()
#guide.get_base_dist()
predictive = Predictive(model, 
                        guide=guide, 
                        num_samples=1000,
                        return_sites=("latent_fairness",))

samples = predictive(data)

posterior_stats = { k : {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        } for k, v in samples.items() }


if False: 
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()
    inferred_mean = alpha_q / (alpha_q + beta_q)
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * np.sqrt(factor)

print("based on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

plt.hist(list(samples.values())[0].flatten().detach().numpy(), 100)

pyro.params.param_store.ParamStoreDict
