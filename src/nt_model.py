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
import pandas as pd
from dataclasses import dataclass
import seabass

def model_base(data,
         guide_std = 1., 
         sigma_noise = 1. 
         ): 
    """ Seabass model for non-targetting guides. 
    
    guide_score ~ Normal(0, guide_std^2)
    log2FC = guide_score [* timepoint] + noise
    noise ~ Normal(0, sigma_noise^2) 
    
    Parameters
    ----------
    Data: a seabass_hier.HierData object. 
    All others are hyperparameters which can be fixed values or distributions, the latter
    if the hyperparameter is being learnt. 
    """

    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 
    guide_std = convertr(guide_std, "guide_std")
    sigma_noise = convertr(sigma_noise, "sigma_noise")
    
    guide_score = pyro.sample("guide_score", 
        dist.Normal(0., guide_std).expand([data.num_guides]).to_event(1)
    )

    mean = guide_score[data.guide_indices]
    if data.multiday: 
        mean *= data.timepoint 
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def fit(data,
       iterations = 1000,
       print_every = 100,
       lr = 0.03,
       sigma_noise = None, # set to None to learn
       learn_guide_std = True): 
    
    one = torch.tensor(1., device = data.device) 
    two = 2. * one
    model = lambda data:  model_base(data, 
         sigma_noise = dist.HalfCauchy(one) if (sigma_noise is None) else sigma_noise, 
         guide_std = dist.HalfCauchy(one) if learn_guide_std else 1. )
    
    to_optimize = ["guide_std",
                  "sigma_noise"]
    
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
    guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))
    
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

