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
         efficacy_prior_a = 1., # shape1 of beta(a,b) prior on guide efficacy
         efficacy_prior_b = 1., # shape2 of beta(a,b) prior on guide efficacy
         junction_std = 1., # junc ~ N(gene, std^2)
         guide_std = 1., 
         sigma_noise = 1., # noise std estimated from non-targetting guides
         sigma_prior = 2.,
         learn_efficacy = True, 
         skew = 1.
         ): 
    """ Seabass model for junctions and genes. 
    
    guide_efficacy ~ Beta(efficacy_prior_a,efficacy_prior_b) for each guide
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

    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 
    sigma_prior = convertr(sigma_prior, "sigma_prior")
    #skew = convertr(skew, "skew")
    efficacy_prior_a = convertr(efficacy_prior_a, "efficacy_prior_a")
    efficacy_prior_b = convertr(efficacy_prior_b, "efficacy_prior_b")
    junction_std = convertr(junction_std, "junction_std")
    guide_std = convertr(guide_std, "guide_std")
    sigma_noise = convertr(sigma_noise, "sigma_noise")
    
    if learn_efficacy: 
        guide_efficacy = pyro.sample("guide_efficacy", 
            dist.Beta(efficacy_prior_a, efficacy_prior_b).expand([data.num_guides]).to_event(1)) 
    else: 
        guide_efficacy = dist.Beta(efficacy_prior_a, efficacy_prior_b).rsample(sample_shape=[data.num_guides])

    gene_essentiality = pyro.sample("gene_essentiality",
        dist.Normal(0., sigma_prior).expand([data.num_genes]).to_event(1)
        #dist.SkewLogistic(0., sigma_prior, skew.exp()).expand([data.num_genes]).to_event(1)  
        #dist.AsymmetricLaplace(0., sigma_prior, skew.exp()).expand([data.num_genes]).to_event(1)   
    )
    
    junction_score = pyro.sample("junction_score", 
        dist.Normal(gene_essentiality[data.junc2gene], junction_std).to_event(1)
    )
    
    guide_score = pyro.sample("guide_score", 
        dist.Normal(junction_score[data.guide2junc], guide_std).to_event(1)
    )

    mean = guide_score[data.guide_indices] * guide_efficacy[data.guide_indices]
    
    if data.multiday: 
        mean *= data.timepoint 
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def fit(data,
       iterations = 1000,
       print_every = 100,
       lr = 0.03,
       learn_sigma = True, 
       sigma_noise = 1., # set to None to learn
       learn_efficacy_prior = True,
       learn_junc_std = True,
       learn_guide_std = True,
       learn_efficacy = True): 
    
    one = torch.tensor(1., device = data.device) 
    two = 2. * one
    model = lambda data:  model_base(data, 
         sigma_prior = dist.HalfCauchy(two) if learn_sigma else 2., 
         sigma_noise = dist.HalfCauchy(one) if (sigma_noise is None) else sigma_noise, 
         efficacy_prior_a = dist.Gamma(two,two) if learn_efficacy_prior else 1., 
         efficacy_prior_b = dist.Gamma(two,two) if learn_efficacy_prior else 1.,
         junction_std = dist.HalfCauchy(one) if learn_junc_std else 1., 
         guide_std = dist.HalfCauchy(one) if learn_guide_std else 1., 
         skew = dist.Normal(0,one),
         learn_efficacy = learn_efficacy
                                    )
    
    to_optimize = ["sigma_prior",
                   "efficacy_prior_a",
                   "efficacy_prior_b",
                  "junction_std",
                   "guide_std",
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

