import seabass
import seabass_hier
import hier_alt
import torch

torch.set_num_threads(20)

import matplotlib.pyplot as plt
import sys
import pyro
import pandas as pd
from pathlib import Path
import numpy as np
from importlib import reload  # Python 3.4+
reload(seabass) 
reload(seabass_hier)
reload(hier_alt)

pyro.clear_param_store()

# load data
data_dir = Path("/gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/")
dat = pd.read_csv(data_dir / "Cas13_essential_arm_foldchanges.txt", sep = "\t")

dat = dat.rename(columns={"Gene": "junction", 
                          "gene.name" : "gene", 
                          "value": "logFC"})
plt.hist(dat.logFC,100)

data = seabass_hier.HierData.from_pandas(dat) 

# for reproducibility
pyro.set_rng_seed(101)

model, guide, losses = hier_alt.fit(data, iterations=3000)

posterior_stats = seabass.get_posterior_stats(model, guide, data) # can alternatively use guide.median()

# check convergence
plt.figure(figsize=(9,4))
plt.plot(losses)
plt.ylabel("ELBO")
plt.xlabel("Iterations")
plt.show()

# plot estimates
plt.figure(figsize=(9,8))
plt.subplot(221)
plt.hist( posterior_stats["guide_efficacy"]["mean"], 30 )
plt.xlabel('guide_efficacy')
plt.subplot(222)
plt.hist( posterior_stats["junction_essentiality"]["mean"], 30 )
plt.xlabel('junction_efficacy')
plt.subplot(223)
plt.hist( posterior_stats["gene_essentiality"]["mean"], 30 )
plt.xlabel('gene_essentiality')
plt.show() 

plt.scatter(posterior_stats["guide_efficacy"]["mean"], 
            posterior_stats["guide_efficacy"]["std"], 
            alpha = 0.05)
plt.xlabel('guide_efficacy mean') 
plt.ylabel('guide_efficacy std') 
plt.show() 

plt.scatter(posterior_stats["junction_efficacy"]["mean"], 
            posterior_stats["junction_efficacy"]["std"], 
            alpha = 0.05)
plt.xlabel('junction_efficacy mean') 
plt.ylabel('junction_efficacy std') 
plt.show() 

