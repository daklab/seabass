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

@dataclass
class HierData(seabass.ScreenData): 
    junction_indices: torch.Tensor
    junctions: pd.Index
    junc2gene: torch.Tensor
    num_junctions: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.num_junctions = max(self.junction_indices) + 1
        
    @staticmethod
    def from_pandas(df, guide_preds = None, device = "cpu"): 
        
        guide_indices, sgrnas = pd.factorize(df.sgrna) # make numeric
        gene_indices, genes = pd.factorize(df.gene)
        junction_indices, junctions = pd.factorize(df.junction)
        
        junc2gene = pd.DataFrame({'junction_indices' : junction_indices, 
                          'gene_indices' : gene_indices}).drop_duplicates()
        assert(np.all(junc2gene.junction_indices == np.arange(junc2gene.shape[0])))
        
        guide_eff = pd.merge( pd.DataFrame( { "sgrna" : sgrnas } ), guide_preds, on = "sgrna", how = "left").guide_eff.fillna(0).values if (not guide_preds is None) else np.zeros( len(sgrnas), dtype = np.float )
        
        if df.week.std() == 0: 
            df.week[:] = 1
        
        return HierData(
            guide_indices = torch.tensor(guide_indices, dtype = torch.long, device = device),
            sgrna_pred = torch.tensor(guide_eff, dtype = torch.float, device = device), 
            junction_indices = torch.tensor(junction_indices, dtype = torch.long, device = device), 
            gene_indices = torch.tensor(gene_indices, dtype = torch.long, device = device), 
            logFC = torch.tensor(np.array(df.logFC), dtype = torch.float, device = device), 
            timepoint = torch.tensor(np.array(df.week), dtype = torch.float, device = device),
            junc2gene = torch.tensor(np.array(junc2gene.gene_indices), dtype = torch.long, device = device), 
            sgrnas = sgrnas, 
            junctions = junctions,
            genes = genes,
            device = device
        )
# model definition 
def model_base(data,
         efficacy_prior_a = 1., # shape1 of beta(a,b) prior on guide efficacy
         efficacy_prior_b = 1., # shape2 of beta(a,b) prior on guide efficacy
         junc_efficacy_prior_a = 1., # shape1 of beta(a,b) prior on guide efficacy
         junc_efficacy_prior_b = 1., # shape2 of beta(a,b) prior on guide efficacy
         sigma_noise = 1., # noise std estimated from non-targetting guides
         sigma_prior = 2.,
         learn_efficacy = True
         ): 
    """ Seabass model for junctions and genes. 
    
    guide_efficacy ~ Beta(efficacy_prior_a,efficacy_prior_b) for each guide
    gene_essentiality ~ Normal(0, sigma_prior^2) for each gene
    junction_efficacy ~ Beta(junc_efficacy_prior_a, junc_efficacy_prior_b): how `targettable` is this junction
    log2FC = gene_essentiality * guide_efficacy * junction_efficacy [* timepoint] + noise
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
    efficacy_prior_a = convertr(efficacy_prior_a, "efficacy_prior_a")
    efficacy_prior_b = convertr(efficacy_prior_b, "efficacy_prior_b")
    junc_efficacy_prior_a = convertr(junc_efficacy_prior_a, "junc_efficacy_prior_a")
    junc_efficacy_prior_b = convertr(junc_efficacy_prior_b, "junc_efficacy_prior_b")
    
    if learn_efficacy: 
        guide_efficacy = pyro.sample("guide_efficacy", 
            dist.Beta(efficacy_prior_a, efficacy_prior_b).expand([data.num_guides]).to_event(1)) 
    else: 
        guide_efficacy = dist.Beta(efficacy_prior_a, efficacy_prior_b).rsample(sample_shape=[data.num_guides])
    
    junction_score = pyro.sample("junction_score", 
        dist.Beta(junc_efficacy_prior_a, junc_efficacy_prior_b).expand([data.num_guides]).to_event(1)
    )

    gene_essentiality = pyro.sample("gene_essentiality",
        dist.Normal(0., sigma_prior).expand([data.num_junctions]).to_event(1)
    )

    mean = gene_essentiality[data.gene_indices] * guide_efficacy[data.guide_indices] * junction_score[data.junction_indices]
    if data.multiday: 
        mean *= data.timepoint 
    with pyro.plate("data", data.guide_indices.shape[0]):
        obs = pyro.sample("obs", dist.Normal(mean, sigma_noise), obs = data.logFC)

def fit(data,
       iterations = 1000,
       print_every = 100,
       lr = 0.03,
       learn_sigma = True, 
       learn_efficacy_prior = True,
       learn_junction_prior = True,
       learn_efficacy = True): 
    two = torch.tensor(2., device = data.device)
    model = lambda data:  model_base(data, 
         sigma_prior = dist.HalfCauchy(two) if learn_sigma else 2., 
         efficacy_prior_a = dist.Gamma(two,two) if learn_efficacy_prior else 1., 
         efficacy_prior_b = dist.Gamma(two,two) if learn_efficacy_prior else 1.,
         junc_efficacy_prior_a = dist.Gamma(two,two) if learn_junction_prior else 1., 
         junc_efficacy_prior_b = dist.Gamma(two,two) if learn_junction_prior else 1.,
         learn_efficacy = learn_efficacy
                                    )
    
    to_optimize = ["sigma_prior",
                   "efficacy_prior_a",
                   "efficacy_prior_b",
                  "junc_efficacy_prior_a",
                  "junc_efficacy_prior_b"]
    
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

