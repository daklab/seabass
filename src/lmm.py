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

def model_base(
    data,
    guide_std = 1., 
    slope_noise = 1., 
    sigma_noise = 1. 
): 
    """ Simpler linear mixed model for guides. 
    
    guide_score ~ Normal(0, guide_std^2) for each guide
    log2FC = (guide_score + guide_random_slope) [* timepoint] + noise
    noise ~ Normal(0, sigma_noise^2) for each observation
    guide_random_slope ~ Normal(0, slope_noise^2) for each (guide,replicate) pair
    
    Parameters
    ----------
    Data: a seabass.Data object. 
    All others are hyperparameters which can be fixed values or distributions, the latter
    if the hyperparameter is being learnt. 
    """

    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 
    
    guide_std = convertr(guide_std, "guide_std")
    sigma_noise = convertr(sigma_noise, "sigma_noise")
    slope_noise = convertr(slope_noise, "slope_noise")
    
    guide_score = pyro.sample("guide_score", 
        dist.Normal(0., guide_std).expand([data.num_guides]).to_event(1)
    )

    mean = guide_score[data.guide_indices]
    if data.multiday: 
        random_slope = pyro.sample("random_slope", # ideally would probably integrate over this
            dist.Normal(0., guide_std).expand([data.num_guides,data.num_replicates]).to_event(2)
        )
        mean = (mean + random_slope[data.guide_indices, data.replicate]) * data.timepoint 
    
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def fit(
    data,
    iterations = 1000,
    print_every = 100,
    lr = 0.03,
    sigma_noise = None, # set to None to learn, or fix to a value
    slope_noise = None,
    guide_std = None
): 
    
    one = torch.tensor(1., device = data.device) 
    two = 2. * one
    model = lambda data:  model_base(
        data, 
        sigma_noise = dist.HalfCauchy(one) if (sigma_noise is None) else sigma_noise, 
        slope_noise = dist.HalfCauchy(one) if (slope_noise is None) else slope_noise,
        guide_std = dist.HalfCauchy(one) if (guide_std is None) else guide_std)
    
    to_optimize = [
        "guide_std",
        "sigma_noise",
        "slope_noise"
    ]
    
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

