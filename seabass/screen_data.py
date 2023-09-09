import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.infer import Predictive
from torch.distributions import constraints

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import seabass

@dataclass
class ScreenData: 
    gene_indices: torch.Tensor # 
    genes: pd.Index
    guide_indices: torch.Tensor
    sgrnas: pd.Index
    guide_to_gene: torch.Tensor
    sgrna_pred: torch.Tensor
    logFC: torch.Tensor
    timepoint: torch.Tensor
    replicate: torch.Tensor
    num_guides: int = field(default = 0, init = False)
    num_genes: int = field(default = 0, init = False)
    multiday: bool = field(default = False , init = False)
    device: torch.device
    
    def __post_init__(self):
        self.num_replicates = max(self.replicate) + 1
        self.num_guides = max(self.guide_indices) + 1
        self.num_genes = max(self.gene_indices) + 1
        self.multiday = self.timepoint.std().item() > 0.
        
    @staticmethod
    def from_pandas(df, guide_preds = None, device = "cpu"): 
        
        guide_indices, sgrnas = pd.factorize(df.sgrna) # make numeric
        gene_indices, genes = pd.factorize(df.gene)
        replicate_indices, replicates = pd.factorize(df.replicate)
        
        guide_to_gene = pd.DataFrame(
            {"guide_indices" : guide_indices,
             "gene_indices" :gene_indices}
        ).drop_duplicates().sort_values("guide_indices")["gene_indices"].to_numpy()
        
        guide_eff = pd.merge( pd.DataFrame( { "sgrna" : sgrnas } ), guide_preds, on = "sgrna", how = "left").guide_eff.fillna(0).values if (not guide_preds is None) else np.zeros( len(sgrnas), dtype = np.float )
        
        if df.week.std() == 0: 
            df.week[:] = 1

        return ScreenData(
            guide_indices = torch.tensor(guide_indices, dtype = torch.long, device = device),
            gene_indices = torch.tensor(gene_indices, dtype = torch.long, device = device), 
            guide_to_gene = torch.tensor(guide_to_gene, dtype = torch.long, device = device), 
            sgrna_pred = torch.tensor(guide_eff, dtype = torch.float, device = device), 
            logFC = torch.tensor(np.array(df.logFC), dtype = torch.float, device = device), 
            timepoint = torch.tensor(np.array(df.week), dtype = torch.float, device = device),
            replicate = torch.tensor(replicate_indices, dtype = torch.long, device = device),
            sgrnas = sgrnas, 
            genes = genes,
            device = device 
        )
