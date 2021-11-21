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

import argparse
import datetime 
parser = argparse.ArgumentParser()

#input file with LFCs
parser.add_argument("input_file")
parser.add_argument("analysis_name")
parser.add_argument("output_dir_name")

args = parser.parse_args()
input_file=args.input_file
analysis_name=args.analysis_name
output_dir=args.output_dir_name

print(input_file)
print(output_dir)
print(analysis_name)

#create directory for output files with date
mydir = os.path.join(output_dir, analysis_name, datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S'))
os.makedirs(mydir,exist_ok=True)
print("Directory '% s' created" % mydir)

#------------------------------------------------------------------
#functions for plotting (to add)
#------------------------------------------------------------------

#fig = my_plot(...)
#fig.savefig("somefile.png")

#------------------------------------------------------------------
#load data (the processing here will eventually be done in the 
#making of the input file)
#input file will be an arguement given to the script 
#------------------------------------------------------------------

#parse arguements to get data input file with log fold changes 
dat = pd.read_csv(input_file, sep = " ")

#keep only essential and common junctions 
dat = dat[dat["junc.type"]=="common"] 
dat = dat[dat["type"]=="essential"] 

#essential gene scores (add file so can look at correlation)

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

#------------------------------------------------------------------
#save results
#------------------------------------------------------------------

#change directory to output and save files there 
os.chdir(mydir)
print("Current working directory: {0}".format(os.getcwd()))

# check convergence
plt.figure(figsize=(9,4))
plt.plot(losses)
plt.ylabel("ELBO")
plt.xlabel("Iterations")
plt.savefig("convergence.png")

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
plt.savefig("estimates.png")

# plot guide mean vs std efficacy 
plt.scatter(posterior_stats["guide_efficacy"]["mean"], 
            posterior_stats["guide_efficacy"]["std"], 
            alpha = 0.05)
plt.xlabel('guide_efficacy mean') 
plt.ylabel('guide_efficacy std') 
plt.savefig("guide_efficacy.png")

# plot junction mean vs std essentiality 
plt.scatter(posterior_stats["junction_essentiality"]["mean"], 
            posterior_stats["junction_essentiality"]["std"], 
            alpha = 0.05)
plt.xlabel('junction_essentiality mean') 
plt.ylabel('junction_essentiality std') 
plt.savefig("junction_essentiality.png")

#save gene essentiality estimates 
x=posterior_stats["gene_essentiality"]["mean"]
px = pd.DataFrame(x.numpy()).T
px["gene"] = data.genes.values
px.columns = ['gene_essentiality', 'gene']

px = px.sort_values(by=['gene_essentiality'])
px.to_csv('gene_essentiality_estimates.csv', index=False)

print("done")