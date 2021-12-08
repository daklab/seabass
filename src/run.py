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
import hier_guide_prior
import torch
import scipy.special

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cpu": torch.set_num_threads(20)

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
reload(hier_guide_prior)
#Clears the global ParamStoreDict . 
#This is especially useful if you're working in a REPL
pyro.clear_param_store()

import argparse
import datetime 
parser = argparse.ArgumentParser()

#input file with LFCs
parser.add_argument("model", choices=['gene', 'hier', 'hier_alt', 'hier_pred'])
parser.add_argument("input_file")
parser.add_argument("analysis_name")
parser.add_argument("output_dir_name")

parser.add_argument("--multiday", action='store_true')

#parser.add_argument("--learn_sigma", action='store_true')
#parser.add_argument("--learn_sigma", action='store_true')
#parser.add_argument("--learn_sigma", action='store_true')

import __main__
is_interactive = not hasattr(__main__, '__file__')
if is_interactive: 
    predictions_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/andrew_model/predictions_wguide_info.txt.gz"
    analysis_name = "R1_include_HW_code"
    #analysis_name = "MAGECK_includeR1"
    analysis_name = "HW_with_pred"
    renamer = None
    if analysis_name == "MAGECK_includeR1": 
        renamer = {"gene" : "junction", "gene.name" : "gene"}
        input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/Cas13_essential_arm_foldchanges_rename.txt"
    else: 
        input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/2021-11-21_HWcode_screen_R1_include/2021-11-21_R1_include_KI_LFC_all_guides_Harm_Code.txt.gz"
    multiday = True
    model_type = "hier_alt"
    output_dir = os.path.expanduser('~/seabass/model_runs/')
else: 
    args = parser.parse_args()
    input_file=args.input_file
    analysis_name=args.analysis_name
    output_dir=args.output_dir_name
    multiday = args.multiday
    model_type = args.model
    
hier = model_type != "gene"

analysis_name += "_" + model_type

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

guide_preds = pd.read_csv(predictions_file, sep = "\t")
guide_preds["guide_eff"] = -(guide_preds.target_lfc - guide_preds.location) / guide_preds.scale # minus! 
guide_preds = guide_preds[ ["sgrna", "guide_eff"] ]

if renamer: dat = dat.rename(columns = renamer) 
#keep only essential and common junctions 
if "junc.type" in dat: dat = dat[dat["junc.type"]=="common"] 
if "type" in dat: dat = dat[dat["type"]=="essential"] 

if not multiday: 
    dat = dat[dat.week==3] 

#essential gene scores (add file so can look at correlation)

# for reproducibility
pyro.set_rng_seed(101)


#------------------------------------------------------------------
#get seabass Hier Data object from input dataset and run model 
#------------------------------------------------------------------
iterations = 3000
if hier: 
    data = seabass_hier.HierData.from_pandas(dat, guide_preds = guide_preds, device = device) 
    if model_type == "hier_alt": 
        model, guide, losses = hier_alt.fit(data, 
                                            iterations=iterations, 
                                            learn_efficacy = True,
                                            learn_efficacy_prior = True) 
    elif model_type == "hier_pred": 
        model, guide, losses = hier_guide_prior.fit(data, iterations=iterations)
    else: 
        model, guide, losses = seabass_hier.fit(data, iterations=iterations)
else: 
    data = seabass.ScreenData.from_pandas(dat, device = device) 
    model, guide, losses = seabass.fit(data, 
                                       iterations=iterations,
                                       learn_efficacy_prior = False,
                                       learn_sigma = True)

posterior_stats = seabass.get_posterior_stats(model, guide, data) # can alternatively use guide.median()

for k,v in posterior_stats.items(): 
    if v['mean'].numel() == 1: print("%s: %.3f ± %.3f" % (k,v['mean'].item(),v['std'].item()))
posterior_stats = {k:{kk:vv.cpu().numpy() for kk,vv in v.items()} for k,v in posterior_stats.items()}

plt.scatter( data.sgrna_pred.cpu(), posterior_stats['guide_efficacy']['mean'], alpha=0.1 )

#plt.scatter( data.sgrna_pred, scipy.special.expit( posterior_stats['guide_efficacy']['mean'] ), alpha=0.1 )
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
ge = posterior_stats["guide_efficacy"]["mean"]
if model_type == "hier_pred": ge = scipy.special.expit(ge)
plt.hist( ge, 30 )
plt.xlabel('guide_efficacy')
if hier: 
    plt.subplot(222)
    plt.hist( posterior_stats["junction_score"]["mean"], 30 )
    plt.xlabel('junction_score')
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
if hier:
    plt.scatter(posterior_stats["junction_score"]["mean"], 
                posterior_stats["junction_score"]["std"], 
                alpha = 0.05)
    plt.xlabel('junction_score mean') 
    plt.ylabel('junction_score std') 
    plt.savefig(results_dir / "junction_score.png")
    plt.show()
    
plt.scatter(posterior_stats["gene_essentiality"]["mean"], 
            posterior_stats["gene_essentiality"]["std"], 
            alpha = 0.2)
plt.xlabel('gene_essentiality mean') 
plt.ylabel('gene_essentiality std') 
plt.savefig(results_dir / "gene_essentiality.pdf")
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
plt.savefig(results_dir / "ge_vs_rna_i.pdf")
plt.show()

rna_i = pd.read_csv("/gpfs/commons/groups/knowles_lab/Cas13Karin/data/A375_public_screens/D2_combined_gene_dep_scores.csv")
rna_i = rna_i.rename({'Unnamed: 0':'gene'}, axis=1)
rna_i = rna_i[ ['gene', 'A375_SKIN'] ]
rna_i.gene = rna_i.gene.str.split(" ", expand=True).iloc[:,0]

merged = pd.merge( rna_i, px, on = "gene" )
plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
scipy.stats.pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )[0]

if hier: 
    je = posterior_stats["junction_score"]["mean"].flatten()
    
    if False: # map mageck data from junctions to genes
        data = seabass_hier.HierData.from_pandas(dat.rename(columns = {"gene" : "junction", "gene.name" : "gene"}))
        je = posterior_stats["gene_essentiality"]["mean"].flatten()
    
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


if False: 
    dat_sim = dat.copy()
    dat_sim.logFC = np.random.normal(scale = 2, size = dat.shape[0])
    data_sim = seabass.ScreenData.from_pandas(dat_sim) 

    # for reproducibility
    pyro.set_rng_seed(101)

    model, guide, losses = seabass.fit(data_sim, iterations=1000)

    posterior_stats = seabass.get_posterior_stats(model, guide, data)

    plt.hist(posterior_stats['gene_essentiality']['mean'],100)

    
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from torch.distributions import constraints

data = 2. * torch.Tensor( [-1., -0.5, -0.5, .5, .8, 1.] )

def model(data):
    guide_efficacy = pyro.sample('guide_efficacy', dist.Beta(1., 1.).expand([len(data)]).to_event(1) )
    gene_essentiality = pyro.sample("gene_essentiality", dist.Normal(0., 5.))
    mean = gene_essentiality * guide_efficacy
    with pyro.plate("data", len(data)):
        obs = pyro.sample("obs", dist.Normal(mean, 1.), obs = data)

def guide(data): 
    prob = pyro.param("prob", torch.tensor(0.5), constraint=constraints.unit_interval)
    z = pyro.sample('assignment', dist.Bernoulli(prob)).long()
    ge_mean = pyro.param("ge_mean", torch.ones(2))
    ge_scale = pyro.param("ge_scale", torch.ones(2), constraint=constraints.positive)
    gene_essentiality = pyro.sample("gene_essentiality", dist.Normal(ge_mean[z], ge_scale[z]))
    guide_efficacy_a = pyro.param('guide_efficacy_a', torch.ones([2,len(data)]), constraint=constraints.positive)
    guide_efficacy_b = pyro.param('guide_efficacy_b', torch.ones([2,len(data)]), constraint=constraints.positive)
    guide_efficacy = pyro.sample("guide_efficacy", dist.Beta(guide_efficacy_a[z,:], guide_efficacy_b[z,:]))  
    return assignment, gene_essentiality, guide_efficacy

TraceEnum_ELBO().loss(model, config_enumerate(guide, "parallel"), data)


#guide = AutoDiagonalNormal(model)
adam = pyro.optim.Adam({"lr" : 0.03})
svi = SVI(model, config_enumerate(guide, "parallel"), adam, loss=TraceEnum_ELBO(max_plate_nesting = 2) ) 

pyro.clear_param_store()
losses = []
for j in range(iterations):
    loss = svi.step(data)
    losses.append(loss)
    if j % 100 == 0: print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))
plt.plot(losses)
ps = seabass.get_posterior_stats(model, guide, data)
ps

pyro.param( 'ge_mean_neg' )
pyro.param( 'ge_mean_pos' )

def model():
    x = pyro.sample('x', dist.Cauchy(3., .5))

#@config_enumerate
def guide(): 
    prob = pyro.param("prob", torch.tensor(0.5), constraint=constraints.unit_interval)
    z = pyro.sample('z', dist.Bernoulli(prob)).long()
    loc = pyro.param("loc", torch.zeros(2))
    scale = pyro.param("scale", .1 * torch.ones(2), constraint=constraints.positive)
    x = pyro.sample('x', dist.Normal(loc[z], scale[z]))
    
adam = pyro.optim.Adam({"lr" : 0.03})
svi = SVI(model, config_enumerate(guide, "parallel"), adam, loss=TraceEnum_ELBO() ) 

pyro.clear_param_store()
losses = []
for j in range(iterations):
    loss = svi.step()
    losses.append(loss)
    if j % 100 == 0: print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))
plt.plot(losses)
pyro.param("loc")
pyro.param("scale")
pyro.param("prob")



from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary

pyro.clear_param_store()

def fit(data,
       learn_sigma = True, 
       sigma_noise = 1., # set to None to learn
       learn_efficacy_prior = True,
       learn_junc_std = True,
       learn_efficacy = True,
       **kwargs): 
    
    one = torch.tensor(1., device = data.device) 
    two = 2. * one
    model = lambda data:  hier_alt.model_base(data, 
         sigma_prior = dist.HalfCauchy(two) if learn_sigma else 2., 
         sigma_noise = dist.HalfCauchy(one) if (sigma_noise is None) else sigma_noise, 
         efficacy_prior_a = dist.Gamma(two,two) if learn_efficacy_prior else 1., 
         efficacy_prior_b = dist.Gamma(two,two) if learn_efficacy_prior else 1.,
         junction_std = dist.HalfCauchy(one) if learn_junc_std else 1., 
         learn_efficacy = learn_efficacy )
                                    
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(nuts_kernel, **kwargs)
    mcmc.run(data)
    return mcmc.get_samples()
    
samples = fit(data, num_samples=1000, warmup_steps=1000)

np_samples = { k:v.cpu().numpy() for k,v in samples.items() }

np.savez("samples.npz", **np_samples)

juncs_fit = pd.DataFrame( {"junction" : data.junctions, 
                           "mcmc" : samples['junction_score'].mean(0).cpu().numpy(),
                           "vb" : posterior_stats['junction_score']['mean'].flatten() } )

sns.scatterplot(x = "mcmc", y = "vb", data = juncs_fit, alpha = 0.1)

guides_fit = pd.DataFrame( {"sgrna" : data.sgrnas, 
                        "mcmc" : samples['guide_efficacy'].mean(0).cpu().numpy(), 
                        "vb" : posterior_stats['guide_efficacy']['mean'].flatten() } )
sns.scatterplot(x = "mcmc", y = "vb", data = guides_fit, alpha = 0.1)

per_junc = dat.groupby('junction').agg({'logFC': 'mean'})
per_junc['abs_logFC'] = np.abs(per_junc.logFC)
per_junc.sort_values(by = "abs_logFC")

juncs_fit.index = juncs_fit.junction

joined = juncs_fit.join(per_junc)
sns.scatterplot(x = "logFC", y = "vb", data = joined, alpha = 0.1)

joined[np.abs(joined.logFC) < 0.1].sort_values(by = "vb")

# JUNC00184507
# JUNC00055534

juncs_fit[juncs_fit.junction == "JUNC00185295"]

junc_0 = dat[dat.junction == "JUNC00185295"]

import seaborn as  sns
junc_0 = pd.merge(junc_0, guides_fit, on="sgrna")
sns.scatterplot(x = "mcmc", y = "vb", hue = "mcmc", data = junc_0)
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)

junc_0["repfactor"], reps = pd.factorize(junc_0.replicate)

plt.figure(figsize=(10,8))
junc_0["week_jitter"] = junc_0.week + junc_0["repfactor"] * 0.15 + np.random.normal(0.,0.01,junc_0.week.shape)
sns.scatterplot(x = "week_jitter", y = "logFC", hue = "mcmc", style = "sgrna", s = 100, x_jitter=400, data = junc_0)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)