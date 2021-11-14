import seabass
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

pyro.clear_param_store()

# load data
data_dir = Path("/gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/")
dat = pd.read_csv(data_dir / "Cas13_essential_arm_foldchanges.txt", sep = "\t")

dat = dat.rename(columns={"Gene": "junction", "value": "logFC"})
plt.hist(dat.logFC,100)

data = seabass.ScreenData.from_pandas(dat) 

# for reproducibility
pyro.set_rng_seed(101)

model, guide, losses = seabass.fit(data, iterations=100)

posterior_stats = seabass.get_posterior_stats(model, guide, data)

# check convergence
plt.figure(figsize=(9,4))
plt.plot(losses)
plt.ylabel("ELBO")
plt.xlabel("Iterations")
plt.show()

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

