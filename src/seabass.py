import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from torch.distributions import constraints

import numpy as np

from dataclasses import dataclass

@dataclass
class OneDay: 
    junction_indices: torch.Tensor
    guide_indices: torch.Tensor
    logFC: torch.Tensor
    num_guides: int = 0
    num_junctions: int = 0
    
    def __post_init__(self):
        self.num_guides = max(self.guide_indices) + 1
        self.num_junctions = max(self.junction_indices) + 1

# model definition 
def model_base(data,
         efficacy_prior_a = 1., # shape1 of beta(a,b) prior on guide efficacy
         efficacy_prior_b = 1., # shape2 of beta(a,b) prior on guide efficacy
         sigma_noise = 1., # noise std estimated from non-targetting guides
         sigma_prior = 2. 
         ): 

    if type(sigma_prior) != float: 
        sigma_prior = pyro.sample("sigma_prior", sigma_prior) # dist.HalfCauchy(torch.tensor(2.)) 
    if type(efficacy_prior_a) != float: 
        efficacy_prior_a = pyro.sample("efficacy_prior_a", efficacy_prior_a) # dist.HalfCauchy(torch.tensor(2.)) 
    if type(efficacy_prior_b) != float: 
        efficacy_prior_b = pyro.sample("efficacy_prior_b", efficacy_prior_b) # dist.HalfCauchy(torch.tensor(2.)) 
    
    #sigma_prior = pyro.sample("sigma_prior", dist.Uniform(0,10) )
    #self.sigma_prior = pyro.param("sigma_prior", torch.tensor(2.), constraint=constraints.positive)
    guide_efficacy = pyro.sample("guide_efficacy", 
        dist.Beta(efficacy_prior_a, efficacy_prior_b).expand([data.num_guides]).to_event(1)
    )

    junction_essentiality = pyro.sample("junction_essentiality",
        dist.Normal(0., sigma_prior).expand([data.num_junctions]).to_event(1)
    )

    mean = junction_essentiality[data.junction_indices] * guide_efficacy[data.guide_indices]
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def get_posterior_stats(model,
                        guide, 
                        data, # OneDay
                        num_samples=100): 
    """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
    guide.requires_grad_(False)
    predictive = Predictive(model, 
                            guide=guide, 
                            num_samples=num_samples,
                            return_sites=("junction_essentiality", 
                                          "guide_efficacy",
                                         "sigma_prior",
                                         "efficacy_prior_a",
                                         "efficacy_prior_b"))

    samples = predictive(data)

    posterior_stats = { k : {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            } for k, v in samples.items() }

    return posterior_stats

def fit(data,
       iterations = 1000,
       print_every = 100,
       lr = 0.03): 
    
    model = lambda data:  model_base(data, 
                                     sigma_prior = dist.HalfCauchy(torch.tensor(2.)), 
                                     efficacy_prior_a = dist.Gamma(torch.tensor(2.),torch.tensor(2.)), 
                                     efficacy_prior_b = dist.Gamma(torch.tensor(2.),torch.tensor(2.)))
    
    to_optimize = ["sigma_prior",
                   "efficacy_prior_a",
                   "efficacy_prior_b"]
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
    guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))
    
    #guide = AutoDiagonalNormal(model)
    
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    # train/fit model
    pyro.clear_param_store()
    losses = []
    for j in range(iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(data)
        losses.append(loss)
        if j % print_every == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data.guide_indices)))
    
    return( model, guide, losses )

