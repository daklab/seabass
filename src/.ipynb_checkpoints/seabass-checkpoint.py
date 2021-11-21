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
from dataclasses import dataclass, field

import seabass

@dataclass
class ScreenData: 
    gene_indices: torch.Tensor
    genes: pd.Index
    guide_indices: torch.Tensor
    sgrnas: pd.Index    
    logFC: torch.Tensor
    timepoint: torch.Tensor
    num_guides: int = field(default = 0, init = False)
    num_genes: int = field(default = 0, init = False)
    multiday: bool = field(default = False , init = False)
    
    def __post_init__(self):
        self.num_guides = max(self.guide_indices) + 1
        self.num_genes = max(self.gene_indices) + 1
        self.multiday = self.timepoint.std().item() > 0.
        
    @staticmethod
    def from_pandas(df): 
        
        guide_indices, sgrnas = pd.factorize(df.sgrna) # make numeric
        gene_indices, genes = pd.factorize(df.gene)

        return ScreenData(
            guide_indices = torch.tensor(guide_indices, dtype = torch.long),
            gene_indices = torch.tensor(gene_indices, dtype = torch.long), 
            logFC = torch.tensor(np.array(df.logFC), dtype = torch.float), 
            timepoint = torch.tensor(np.array(df.week), dtype = torch.float),
            sgrnas = sgrnas, 
            genes = genes
        )

# model definition 
def model_base(data,
         efficacy_prior_a = 1., # shape1 of beta(a,b) prior on guide efficacy
         efficacy_prior_b = 1., # shape2 of beta(a,b) prior on guide efficacy
         sigma_noise = 1., # noise std estimated from non-targetting guides
         sigma_prior = 2. 
         ): 
    """ Seabass model for genes (or junctions). 
    
    guide_efficacy ~ Beta(efficacy_prior_a,efficacy_prior_b) for each guide
    gene_essentiality ~ Normal(0, sigma_prior^2) for each gene
    log2FC = gene_essentiality * guide_efficacy [* timepoint] + noise
    noise ~ Normal(0, sigma_noise^2) 
    
    Parameters
    ----------
    Data: a seabass_hier.HierData object. 
    All others are hyperparameters which can be fixed values or distributions, the latter
    if the hyperparameter is being learnt. 
    """
    
    if type(sigma_prior) != float: 
        sigma_prior = pyro.sample("sigma_prior", sigma_prior)
    if type(efficacy_prior_a) != float: 
        efficacy_prior_a = pyro.sample("efficacy_prior_a", efficacy_prior_a)
    if type(efficacy_prior_b) != float: 
        efficacy_prior_b = pyro.sample("efficacy_prior_b", efficacy_prior_b)
        
    guide_efficacy = pyro.sample("guide_efficacy", 
        dist.Beta(efficacy_prior_a, efficacy_prior_b).expand([data.num_guides]).to_event(1)
    )

    gene_essentiality = pyro.sample("gene_essentiality",
        dist.Normal(0., sigma_prior).expand([data.num_genes]).to_event(1)
    )

    mean = gene_essentiality[data.gene_indices] * guide_efficacy[data.guide_indices]
    if data.multiday: 
        mean *= data.timepoint 
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def get_posterior_stats(model,
                        guide, 
                        data, 
                        num_samples=100): 
    """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
    guide.requires_grad_(False)
    predictive = Predictive(model, 
                            guide=guide, 
                            num_samples=num_samples) 

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
       lr = 0.03,
       learn_sigma = True, 
       learn_efficacy_prior = True): 
    
    model = lambda data:  model_base(data, 
         sigma_prior = dist.HalfCauchy(torch.tensor(2.)) if learn_sigma else 2., 
         efficacy_prior_a = dist.Gamma(torch.tensor(2.),torch.tensor(2.)) if learn_efficacy_prior else 1., 
         efficacy_prior_b = dist.Gamma(torch.tensor(2.),torch.tensor(2.)) if learn_efficacy_prior else 1.
                                    )
    
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

