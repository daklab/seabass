import seabass
import torch
import matplotlib.pyplot as plt
import sys
import pyro
import pandas as pd
from pathlib import Path
import numpy as np
from importlib import reload  # Python 3.4+
reload(seabass) 

# load data
data_dir = Path("/gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/")
dat = pd.read_csv(data_dir / "Cas13_essential_arm_foldchanges.txt", sep = "\t")
day21 = dat[dat.day == "day21"].rename(columns={"Gene": "junction", "value": "logFC"})
day21.iloc[:,[0,1,2,-2]]
plt.hist(day21.logFC,100)

guide_indices, sgrnas = pd.factorize(day21.sgrna) # make numeric
junction_indices, juncs = pd.factorize(day21.junction)

# convert to Torch
data = seabass.OneDay(
    guide_indices = torch.tensor(guide_indices, dtype = torch.long),
    junction_indices = torch.tensor(junction_indices, dtype = torch.long), 
    logFC = torch.tensor(np.array(day21.logFC), dtype = torch.float)
)

# for reproducibility
pyro.set_rng_seed(101)

model, guide, losses = seabass.fit(data, iterations=101)

posterior_stats = model.get_posterior_stats(guide, data)

# check convergence
plt.figure(figsize=(9,4))
plt.subplot(121)
plt.plot(losses)
plt.ylabel("ELBO")
plt.xlabel("Iterations")
plt.subplot(122)
plt.plot(np.arange(1000,5000), losses[1000:]) # seems converged after ~2000 
plt.xlabel("Iterations")


# plot estimates
plt.figure(figsize=(9,4))
plt.subplot(121)
plt.hist( posterior_stats["guide_efficacy"]["mean"], 30 )
plt.xlabel('guide_efficacy')
plt.subplot(122)
plt.hist( posterior_stats["junction_essentiality"]["mean"], 30 )
plt.xlabel('junction_essentiality')
plt.show() 

plt.scatter(posterior_stats["guide_efficacy"]["mean"], 
            posterior_stats["guide_efficacy"]["std"], 
            alpha = 0.05)
plt.xlabel('guide_efficacy mean') 
plt.ylabel('guide_efficacy std') 
plt.show() 

plt.scatter(posterior_stats["junction_essentiality"]["mean"], 
            posterior_stats["junction_essentiality"]["std"], 
            alpha = 0.05)
plt.xlabel('junction_essentiality mean') 
plt.ylabel('junction_essentiality std') 
plt.show() 

# TODO
# - multiple days
# - deal with bimodality: are these just 0s? 
# - learn prior on efficacy? (mixture?) 
# - learn prior on essentiality? 
# - should prior on essentiality have negative mean? (in essential arm) 
# - hierarchy over genes/junctions
#  - version 1: lfc = guide_efficacy * junction_targetability * gene_essentiality 
#  - version 2: junction_essentiality ~ N( gene_essentiality, sigma2 )