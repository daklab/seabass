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
         junction_std = 1., # junc ~ N(gene, std^2)
         sigma_noise = 1., # noise std estimated from non-targetting guides
         sigma_prior = 2. 
         ): 
    """ Seabass model for junctions and genes. 
    
    guide_efficacy ~ Normal(a * predicted_eff + b, efficacy_std) for each guide
    gene_essentiality ~ Normal(0, sigma_prior^2) for each gene
    junction_essentiality ~ Normal(gene_essentiality, junction_std^2) matching junctions to genes
    log2FC = junction_essentiality * guide_efficacy [* timepoint] + noise
    noise ~ Normal(0, sigma_noise^2) 
    
    Parameters
    ----------
    Data: a seabass_hier.HierData object. 
    All others are hyperparameters which can be fixed values or distributions, the latter
    if the hyperparameter is being learnt. 
    """

    if type(sigma_prior) != float: 
        sigma_prior = pyro.sample("sigma_prior", sigma_prior)
    if type(junction_std) != float: 
        junction_std = pyro.sample("junction_std", junction_std)
        
    guide_a = pyro.sample("guide_a", dist.Normal(1, 1).to_event(1) )
    guide_b = pyro.sample("guide_b", dist.Normal(0, 1).to_event(1) )
    efficacy_std = pyro.sample("efficacy_std", dist.HalfCauchy(torch.tensor(2.)).to_event(1) )
    
    guide_efficacy = pyro.sample("guide_efficacy", 
        dist.Normal(guide_a * data.predicted_guide_eff + guide_b, efficacy_std).to_event(1)
    )
    
    gene_essentiality = pyro.sample("gene_essentiality",
        dist.Normal(0., sigma_prior).expand([data.num_genes]).to_event(1)
    )
    
    junction_score = pyro.sample("junction_score", 
        dist.Normal(gene_essentiality[data.junc2gene], junction_std).to_event(1)
    )

    mean = junction_score[data.junction_indices] * torch.logit(guide_efficacy[data.guide_indices])
    if data.multiday: 
        mean *= data.timepoint 
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def fit(data,
       iterations = 1000,
       print_every = 100,
       lr = 0.03,
       learn_sigma = True, 
       learn_junc_std = True): 
    
    model = lambda data:  model_base(data, 
         sigma_prior = dist.HalfCauchy(torch.tensor(2.)) if learn_sigma else 2., 
         junction_essentiality_std = dist.HalfCauchy(torch.tensor(1.)) if learn_junc_std else 1., 
                                    )
    to_optimize = ["sigma_prior",
                   "guide_a",
                   "guide_b",
                   "efficacy_std",
                  "junction_std"]
    
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

