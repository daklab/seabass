#------------------------------------------------------------------
#SETUP
#------------------------------------------------------------------

#set working directory to git folder on the cluster 
import os
os.chdir(os.path.expanduser('~/seabass/src/'))

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

parser.add_argument("--multiday", action='store_true')

import __main__
is_interactive = not hasattr(__main__, '__file__')
if is_interactive: 
    input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/2021-11-21_HWcode_screen_R1_include/2021-11-21_R1_include_KI_LFC_all_guides_Harm_Code.txt.gz"
    analysis_name = "R1_include_HW_code"
    multiday = False
    output_dir = os.path.expanduser('~/seabass/model_runs/')
else: 
    args = parser.parse_args()
    input_file=args.input_file
    analysis_name=args.analysis_name
    output_dir=args.output_dir_name
    multiday = args.multiday

analysis_name += "_multiday" if multiday else "_day21"
print(input_file)
print(output_dir)
print(analysis_name)

#create directory for output files with date
results_dir = Path(output_dir) / analysis_name / datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
results_dir.mkdir(parents=True, exist_ok=True)

print("Directory '% s' created" % results_dir)

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

if not multiday: 
    dat = dat[dat.week==3] 

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

# check convergence
plt.figure(figsize=(5,4))
plt.plot(losses)
plt.ylabel("ELBO")
plt.xlabel("Iterations")
plt.savefig(results_dir / "convergence.png")
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
plt.savefig(results_dir / "estimates.png")
plt.show()

# plot guide mean vs std efficacy 
plt.scatter(posterior_stats["guide_efficacy"]["mean"], 
            posterior_stats["guide_efficacy"]["std"], 
            alpha = 0.05)
plt.xlabel('guide_efficacy mean') 
plt.ylabel('guide_efficacy std') 
plt.savefig(results_dir / "guide_efficacy.png")
plt.show()

# plot junction mean vs std essentiality 
plt.scatter(posterior_stats["junction_essentiality"]["mean"], 
            posterior_stats["junction_essentiality"]["std"], 
            alpha = 0.05)
plt.xlabel('junction_essentiality mean') 
plt.ylabel('junction_essentiality std') 
plt.savefig(results_dir / "junction_essentiality.png")
plt.show()

#save gene essentiality estimates 
ge = posterior_stats["gene_essentiality"]["mean"].numpy().flatten()
px = pd.DataFrame({"gene_essentiality" : ge, "gene" : data.genes.values})

px = px.sort_values(by=['gene_essentiality'])
px.to_csv(results_dir / 'gene_essentiality_estimates.csv', index=False)

gecko = pd.read_csv("/gpfs/commons/home/mschertzer/cas_library/achilles_geckoV2_19Q4.csv.gz")
gecko = gecko.rename({'Unnamed: 0':'gene'}, axis=1)
gecko = gecko[ ['gene', 'A375_SKIN'] ]

merged = pd.merge( gecko, px, on = "gene" )
plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
import scipy.stats
scipy.stats.pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )[0]

rna_i = pd.read_csv("/gpfs/commons/groups/knowles_lab/Cas13Karin/data/A375_public_screens/D2_combined_gene_dep_scores.csv")
rna_i = rna_i.rename({'Unnamed: 0':'gene'}, axis=1)
rna_i = rna_i[ ['gene', 'A375_SKIN'] ]
rna_i.gene = rna_i.gene.str.split(" ", expand=True).iloc[:,0]

merged = pd.merge( rna_i, px, on = "gene" )
plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
scipy.stats.pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )[0]

je = posterior_stats["junction_essentiality"]["mean"].flatten()

num_genes = data.junc2gene.max()+1
gene_scores = {
    "max_je" : np.array( [ je.flatten()[data.junc2gene == i].max().item() for i in np.arange(num_genes) ] ), 
    "min_je" : np.array([ je.flatten()[data.junc2gene == i].min().item() for i in np.arange(num_genes) ]), 
    "mean_je" : np.array([ je.flatten()[data.junc2gene == i].mean().item() for i in np.arange(num_genes) ]), 
    "ge" : ge 
}

plt.figure(figsize=(9,8))
for i,(k,v) in enumerate(gene_scores.items()): 
    px = pd.DataFrame( {"gene_essentiality" : v, "gene" : data.genes.values} )
    merged = pd.merge( rna_i, px, on = "gene" )
    plt.subplot(2,2,i+1)
    plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
    r,_ = scipy.stats.pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )
    plt.title("%s Pearson R=%.3f" % (k, r))
    if i>=2 : plt.xlabel("RNAi")
    if i%2 == 0: plt.ylabel("Cas13")
plt.savefig(results_dir / "vs_rna_i.pdf")
plt.show()

print("done")
