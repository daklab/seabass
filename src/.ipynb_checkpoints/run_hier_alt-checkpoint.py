#------------------------------------------------------------------
#SETUP
#------------------------------------------------------------------

#set working directory to git folder on the cluster 
import os
os.chdir('/gpfs/commons/home/kisaev/seabass/src/')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

#import libraries and models 
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

#Clears the global ParamStoreDict . 
#This is especially useful if you're working in a REPL
pyro.clear_param_store()

#------------------------------------------------------------------
#functions for plotting  
#------------------------------------------------------------------

#def my_plot(location_dataset, min_temperature, max_temperature):
#    condition = (location_dataset['temperature'] > min_temperature) & (dataset['temperature'] <= max_temperature)
#    subset = location_dataset[condition] # subset the data based on the temperature range

#    x = subset['precipitation'] # takes the precipitation column only
    # N.B. referenca taken to fig
#    fig = plt.figure(figsize=(8, 6))
#    plt.plot(x)
#    plt.show()

#    return fig
#fig = my_plot(...)
#fig.savefig("somefile.png")

#------------------------------------------------------------------
#load data (the processing here will eventually be done in the 
#making of the input file)
#input file will be an arguement given to the script 
#------------------------------------------------------------------

#parse arguements to get data input file with log fold changes 
data_dir = Path("/gpfs/commons/groups/knowles_lab/Cas13Karin/data/")
dat = pd.read_csv(data_dir / "2021-11-19_KI_LFC_all_guides_Harm_Code.txt.gz", sep = " ")

#rename column names so they match seabass data object 
dat = dat.rename(columns={"junc.name": "junction", 
                          "guide.id" : "sgrna",
                          "gene.name" : "gene", 
                          "value": "logFC"})

#keep only essential and common junctions 
dat = dat[dat["junc.type"]=="common"] 
dat = dat[dat["type"]=="essential"] 

#make week numerical value 
dat["week"]=1 
dat["week"][dat["day"]=="D14"] = 2
dat["week"][dat["day"]=="D21"] = 3

#------------------------------------------------------------------
#get seabass Hier Data object from input dataset 
#------------------------------------------------------------------

data = seabass_hier.HierData.from_pandas(dat) 

#------------------------------------------------------------------
#run model 
#------------------------------------------------------------------

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

