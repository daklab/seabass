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
import lmm
import seabass_hier
import hier_alt
import hier_hier
import hier_guide_prior
import hier_usage
import nt_model
import torch
import scipy.special

import scipy.stats

def nan_pearsonr(x,y): 
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return scipy.stats.pearsonr(x[~nas], y[~nas])


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
reload(lmm)
reload(seabass_hier)
reload(hier_alt)
reload(hier_hier)
reload(nt_model)
reload(hier_guide_prior)
reload(hier_usage)
#Clears the global ParamStoreDict . 
#This is especially useful if you're working in a REPL
pyro.clear_param_store()

import argparse
import datetime 
parser = argparse.ArgumentParser()

#input file with LFCs
parser.add_argument("model", choices=['gene', 'hier', 'hier_alt', 'hier_pred', 'hier_hier', 'nt_model', 'hier_usage'])
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
    #analysis_name = "R1_include_HW_code"
    #analysis_name = "MAGECK_includeR1"
    analysis_name = "FinalData"
    renamer = None
    if analysis_name == "MAGECK_includeR1": 
        renamer = {"gene" : "junction", "gene.name" : "gene"}
        input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/analysis/Cas13_essential_arm_foldchanges_rename.txt"
    else: 
        #input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/2021-11-21_HWcode_screen_R1_include/2021-11-21_R1_include_KI_LFC_all_guides_Harm_Code.txt.gz"
        #input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/screen_processing/2021-12-09_HWcode_screen_R1_include/2021-12-09_R1_include_normalized_wbatch_cor_KI_LFC_all_guides_Harm_Code.txt.gz"
        #input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/screen_processing/2021-12-09_HWcode_screen_R1_include/2021-12-09_R1_include_normalized_wnobatch_cor_KI_LFC_all_guides_Harm_Code.txt.gz"
        #input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/screen_processing/2021-12-09_HWcode_screen_R1_remove/2021-12-09_R1_remove_normalized_wbatch_cor_KI_LFC_all_guides_Harm_Code.txt.gz"
        #input_file = "/gpfs/commons/groups/knowles_lab/Cas13Karin/data/screen_processing/2021-12-09_HWcode_screen_R1_remove/2021-12-09_R1_remove_normalized_wnobatch_cor_KI_LFC_all_guides_Harm_Code.txt.gz"
        #input_file = "/gpfs/commons/groups/knowles_lab/cas13_share/data/2022-01-22_all_reps/tech_reps_pooled/2022-01-22_R1_removeTechPool-normalized_wnobatch_corMS_LFC_all_guides_Harm_Code.txt.gz"
        input_file = "/gpfs/commons/groups/knowles_lab/cas13_share/data/2022-02-23_all_reps/2022-02-23_R1_remove_TechPool-normalized_wnobatch_corMS_LFC_all_guides_Harm_Code.txt.gz"
        
    multiday = True
    #model_type = "hier_hier"
    model_type = "NT"
    hier= False
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
dat = pd.read_csv(input_file, sep = "\t")

guide_preds = pd.read_csv(predictions_file, sep = "\t")
guide_preds["guide_eff"] = -(guide_preds.target_lfc - guide_preds.location) / guide_preds.scale # minus! 
guide_preds = guide_preds[ ["sgrna", "guide_eff"] ]

if renamer: dat = dat.rename(columns = renamer) 
#keep only essential and common junctions 
if model_type == "nt_model": 
    dat = dat[dat["type"]=="NT"] 
else:
    if model_type != "hier_usage":
        if "junc.type" in dat: dat = dat[dat["junc.type"]=="common"] # choice of these should be another setting
        if "type" in dat: dat = dat[dat["type"]=="essential"] 
    else: 
        if "type" in dat: dat = dat[dat["type"]!="NT"] 
    
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
    elif model_type == "hier_hier": 
        model, guide, losses = hier_hier.fit(data, iterations=iterations, sigma_noise = None, learn_efficacy_prior = False)
    elif model_type == "nt_model": 
        model, guide, losses = nt_model.fit(data, iterations=iterations, sigma_noise = None)
    elif model_type == "hier_usage": 
        model, guide, losses = hier_usage.fit(data, iterations=iterations, sigma_noise = None, learn_efficacy_prior = False)
    elif model_type == "hier": 
        model, guide, losses = seabass_hier.fit(data, iterations=iterations)
    else: 
        raise Exception("Unknown model type") 
else: 
    assert(model_type == "gene")
    data = seabass.ScreenData.from_pandas(dat, device = device) 
    model, guide, losses = seabass.fit(
        data, 
        iterations=iterations,
        learn_efficacy_prior = False,
        learn_sigma = True)

import lmm_hier
reload(lmm_hier)
model, guide, losses = lmm_hier.fit(data, iterations=3000)

posterior_stats = seabass.get_posterior_stats(model, guide, data) # can alternatively use guide.median()

for k,v in posterior_stats.items(): 
    if v['mean'].numel() == 1: print("%s: %.3f ± %.3f" % (k,v['mean'].item(),v['std'].item()))
posterior_stats = {k:{kk:vv.cpu().numpy().squeeze() for kk,vv in v.items()} for k,v in posterior_stats.items()}

plt.figure(figsize=(5,4))
plt.scatter( data.sgrna_pred.cpu(), posterior_stats['guide_efficacy']['mean'], alpha=0.1 )
plt.xlabel("Predicted efficacy (from sequence)")
plt.ylabel("Estimated efficacy")
plt.savefig(results_dir / "guide_efficacy_vs_pred.pdf")
plt.show()
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
ge = posterior_stats["gene_essentiality"]["mean"]
px = pd.DataFrame({"gene_essentiality" : ge, "gene" : data.genes.values})

px = px.sort_values(by=['gene_essentiality'])
px.to_csv(results_dir / 'gene_essentiality_estimates.csv', index=False)

gecko = pd.read_csv("/gpfs/commons/home/mschertzer/cas_library/achilles_geckoV2_19Q4.csv.gz")
gecko = gecko.rename({'Unnamed: 0':'gene'}, axis=1)
gecko = gecko[ ['gene', 'A375_SKIN'] ]

merged = pd.merge( gecko, px, on = "gene" )
plt.figure(figsize = (5,4))
plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
plt.xlabel("Gecko")
plt.ylabel("Cas13")
r,_ = nan_pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )
plt.title("Pearson R=%.3f" % r)        
plt.show()

rna_i = pd.read_csv("/gpfs/commons/groups/knowles_lab/Cas13Karin/data/A375_public_screens/D2_combined_gene_dep_scores.csv")
rna_i = rna_i.rename({'Unnamed: 0':'gene'}, axis=1)
rna_i = rna_i[ ['gene', 'A375_SKIN'] ]
rna_i.gene = rna_i.gene.str.split(" ", expand=True).iloc[:,0]

merged = pd.merge( rna_i, px, on = "gene" )
plt.figure(figsize = (5,4))
plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
plt.xlabel("RNAi")
plt.ylabel("Cas13")
r,_ = nan_pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )
plt.title("Pearson R=%.3f" % r) 
plt.savefig(results_dir / "ge_vs_rna_i.pdf")
plt.show()

if hier: 
    je = posterior_stats["junction_score"]["mean"].flatten()
    
    if False: # map mageck data from junctions to genes
        data = seabass_hier.HierData.from_pandas(dat.rename(columns = {"gene" : "junction", "gene.name" : "gene"}))
        je = posterior_stats["gene_essentiality"]["mean"].flatten()
    
    junc2gene = data.junc2gene.cpu().numpy()
    num_genes = junc2gene.max()+1
     
    gene_scores = {
        "max_je" : np.array( [ je[junc2gene == i].max() for i in np.arange(num_genes) ] ), 
        "min_je" : np.array([ je[junc2gene == i].min() for i in np.arange(num_genes) ]), 
        "mean_je" : np.array([ je[junc2gene == i].mean() for i in np.arange(num_genes) ]), 
        "ge" : ge 
    }

    plt.figure(figsize=(9,8))
    for i,(k,v) in enumerate(gene_scores.items()): 
        px = pd.DataFrame( {"gene_essentiality" : v, "gene" : data.genes.values} )
        merged = pd.merge( rna_i, px, on = "gene" )
        plt.subplot(2,2,i+1)
        plt.scatter(merged['A375_SKIN'], merged['gene_essentiality'])
        r,_ = nan_pearsonr( merged['A375_SKIN'], merged['gene_essentiality'] )
        plt.title("%s Pearson R=%.3f" % (k, r))
        if i>=2 : plt.xlabel("RNAi")
        if i%2 == 0: plt.ylabel("Cas13")
    plt.savefig(results_dir / "vs_rna_i.pdf")
    plt.show()

print("done")
stop()

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

#np.savez("samples.npz", **np_samples)
np_samples = np.load("samples.npz")
import seaborn as  sns

juncs_fit = pd.DataFrame( {"junction" : data.junctions, 
                           "mcmc" : np_samples['junction_score'].mean(0),
                           "vb" : posterior_stats['junction_score']['mean'].flatten() } )

sns.scatterplot(x = "mcmc", y = "vb", data = juncs_fit, alpha = 0.1)
plt.show() 

guides_fit = pd.DataFrame( {"sgrna" : data.sgrnas, 
                        "mcmc" : np_samples['guide_efficacy'].mean(0), 
                        "vb" : posterior_stats['guide_efficacy']['mean'].flatten() } )
sns.scatterplot(x = "mcmc", y = "vb", data = guides_fit, alpha = 0.1)
plt.show() 

per_junc = dat.groupby('junction').agg({'logFC': 'mean'})
per_junc['abs_logFC'] = np.abs(per_junc.logFC)
per_junc.sort_values(by = "abs_logFC")

juncs_fit.index = juncs_fit.junction

joined = juncs_fit.join(per_junc)

sns.scatterplot(x = "logFC", y = "vb", data = joined, alpha = 0.1)
plt.show() 

joined[np.abs(joined.logFC) < 0.1].sort_values(by = "vb")

# JUNC00184507
# JUNC00055534
# JUNC00185295
juncs_fit[juncs_fit.junction == "JUNC00055534"]

junc_0 = dat[dat.junction == "JUNC00055534"]


junc_0 = pd.merge(junc_0, guides_fit, on="sgrna")
sns.scatterplot(x = "mcmc", y = "vb", hue = "mcmc", data = junc_0)
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
plt.show() 

junc_0["repfactor"], reps = pd.factorize(junc_0.replicate)

plt.figure(figsize=(8,6))
junc_0["week_rep"] = junc_0.week + junc_0["repfactor"] * 0.15 + np.random.normal(0.,0.01,junc_0.week.shape)
sns.scatterplot(x = "week_rep", y = "logFC", hue = "efficacy", style = "sgrna", s = 100, x_jitter=400, data = junc_0.rename(columns={"mcmc" : "efficacy"}))
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xlabel("week + 0.15 * replicate")
plt.show() 


## junction specific effects
gene_essentiality = posterior_stats['gene_essentiality']['mean']
junc_prior_mean = gene_essentiality[data.junc2gene.cpu()] * data.perc_usage.cpu().numpy()

#z = (posterior_stats['junction_score']['mean'] - junc_prior_mean) / posterior_stats['junction_score']['std']

z = posterior_stats['junction_score']['mean'] / posterior_stats['junction_score']['std']

junc_i = np.argmax(z)
gene_i = data.junc2gene[junc_i].item()
junc_name = data.junctions[junc_i]
gene_name = data.genes[gene_i]

this_gene = dat[dat.gene == gene_name]

plt.figure(figsize=(8,6))
this_gene["repfactor"], reps = pd.factorize(this_gene.replicate)
this_gene["week_rep"] = this_gene.week + this_gene["repfactor"] * 0.15 + np.random.normal(0.,0.01,this_gene.week.shape)
sns.scatterplot(x = "week_rep", y = "logFC", hue = "junction", s = 100, x_jitter=400, data = this_gene)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xlabel("week + 0.15 * replicate")
plt.show()

junc2cluster = pd.read_csv("/gpfs/commons/home/mschertzer/cas13-isoforms/all_replicates/Data/211215_cluster_id.txt", sep="\t")
#junc2cluster.index= junc2cluster["junc.name"]
#junc2cluster.loc[data.junctions.values,:]
#junc2cluster.loc['JUNC00261806',:]

merged = pd.DataFrame({"junc.name" : data.junctions.values, 
                       "m" : posterior_stats['junction_score']['mean'], 
                       "s" : posterior_stats['junction_score']['std'], 
                       "perc_usage" : data.perc_usage.cpu() }).merge(junc2cluster, on = "junc.name")

set(data.junctions.values).intersection(set(junc2cluster["junc.name"].values))

clusters = merged["cluster.n"].unique()
res = []
for c in clusters: 
    sub = merged[merged["cluster.n"] == c].reset_index()
    for i in range(1,sub.shape[0]):
        for j in range(i):
            res.append([ sub["junc.name"][i],
                        sub["junc.name"][j],
                        sub.m[i],
                        sub.s[i],
                        sub.m[j],
                        sub.s[j],
                        sub.perc_usage[i],
                        sub.perc_usage[j] ] )
            
diff = pd.DataFrame(res, columns = ["j1", "j2", "m1", "s1", "m2", "s2", "perc1", "perc2"])
diff["z1"] = diff.m1 / diff.s1
diff["z2"] = diff.m2 / diff.s2

diff["p1"] = scipy.stats.norm.cdf(np.abs(diff.z1))
diff["p2"] = scipy.stats.norm.cdf(np.abs(diff.z2))
diff["pp"] = diff.p1 * diff.p2

diff["zz"] = diff.z1 * diff.z2
diff = diff.sort_values("zz").reset_index()

junc2gene = dat[["junction","gene","type"]].drop_duplicates()
diff = diff.merge(junc2gene, left_on = "j1", right_on = "junction")

diff_neg = diff[diff.zz < -1.]
diff_neg = diff_neg.sort_values("pp", ascending = False).reset_index()


diff_sig = diff_neg[diff_neg.pp > .9].copy()
diff_sig["pos_z"] = np.where(diff_sig.m1 > 0, diff_sig.z1, diff_sig.z2)
diff_sig = diff_sig.sort_values("pos_z", ascending = False)


i = 168

junc_1 = diff_neg.j1[i]
junc_2 = diff_neg.j2[i]

junc_i = np.where(data.junctions.values == junc_1)[0][0]
gene_i = data.junc2gene[junc_i].item()
gene_name = data.genes[gene_i]

this_gene = dat[dat.junction.isin([junc_1,junc_2])].copy()

plt.figure(figsize=(8,6))
this_gene["repfactor"], reps = pd.factorize(this_gene.replicate)
this_gene["week_rep"] = this_gene.week + this_gene["repfactor"] * 0.15 + np.random.normal(0.,0.01,this_gene.week.shape)
sns.scatterplot(x = "week_rep", y = "logFC", hue = "junction", s = 100, x_jitter=400, data = this_gene)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xlabel("week + 0.15 * replicate")
plt.title(gene_name)
plt.show()

diff_neg.drop(['level_0','index','junction'], axis=1, inplace=True)

diff_neg.to_csv('divergent_junc.csv')





import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.transforms import affine_autoregressive, iterated
from pyro.distributions.util import eye_like, sum_rightmost
from pyro.nn.module import PyroModule, PyroParam


pyro.clear_param_store() 

# want N PxP MVNs
N = 5
P = 3 
init_scale = 0.1
loc = pyro.param("loc",torch.zeros(N,P))

scale = pyro.param(
    "scale",
    torch.full_like(loc, init_scale), 
    constraint = constraints.softplus_positive
)

scale_tril = pyro.param(
    "scale_tril",
    eye_like(loc, P).repeat([N,1,1]), 
    constraint = constraints.unit_lower_cholesky
)

unscale_tril = scale[...,None] * scale_tril
post = dist.MultivariateNormal(loc, scale_tril=unscale_tril)


post.sample()
