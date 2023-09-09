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

# model definition 
class Seabass(PyroModule):
    
    def __init__(self, 
                 num_guides, 
                 num_junctions,
                 efficacy_prior_a = 1., # shape1 of beta(a,b) prior on guide efficacy
                 efficacy_prior_b = 1., # shape2 of beta(a,b) prior on guide efficacy
                 sigma_noise = 1., # noise std estimated from non-targetting guides
                 sigma_prior = 2. 
                ): 
        """ Setting up parameters and priors """
        super().__init__()
        #self.sigma_prior = PyroSample( dist.HalfCauchy(torch.tensor(sigma_prior)).to_event() )
        self.sigma_prior = PyroSample( dist.Uniform(0,10) )
        #self.sigma_prior = pyro.param("sigma_prior", torch.tensor(2.), constraint=constraints.positive)
        self.num_junctions = num_junctions
        self.guide_efficacy = PyroSample(
            dist.Beta(efficacy_prior_a, efficacy_prior_b).expand([num_guides]).to_event(1)
        )
        self.sigma_noise = sigma_noise

    def forward(self, data):
        """ The generative process/model """
        
        self.junction_essentiality = PyroSample(
            dist.Normal(0., self.sigma_prior).expand([self.num_junctions]).to_event(1)
        )
            
        mean = self.junction_essentiality[data.junction_indices] * self.guide_efficacy[data.guide_indices]
        with pyro.plate("data", data.guide_indices.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, self.sigma_noise), obs = data.logFC)
        return mean # can we return nothing? 

    def get_posterior_stats(self,
                            guide, 
                            data, # OneDay
                            num_samples=100): 
        """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
        guide.requires_grad_(False)
        predictive = Predictive(self, 
                                guide=guide, 
                                num_samples=num_samples,
                                return_sites=("junction_essentiality", 
                                              "guide_efficacy",
                                             "sigma_prior"))

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
    
    num_guides = max(data.guide_indices) + 1
    num_junctions = max(data.junction_indices) + 1
    
    # create model
    model = Seabass( num_guides = num_guides, 
                     num_junctions = num_junctions )
    
    #guide = AutoGuideList(model)
    #guide.add(AutoDiagonalNormal(poutine.block(model, hide=["sigma_prior"])))
    #guide.add(AutoDelta(poutine.block(model, expose=["sigma_prior"])))
    
    guide = AutoDiagonalNormal(model)
    
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

