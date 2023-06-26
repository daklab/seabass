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
import torch
import scipy.special
import scipy.stats
import plotnine as p9
import pickle


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
import argparse
import datetime 

from importlib import reload  # Python 3.4+

#Clears the global ParamStoreDict . 
#This is especially useful if you're working in a REPL
pyro.clear_param_store()

analysis_name = "LMM"
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
model_type = "nt_model"
hier= False
output_dir = os.path.expanduser('~/seabass/model_runs/')

analysis_name += "_" + model_type

analysis_name += "_multiday" if multiday else "_day21"

#create directory for output files with date
results_dir = Path(output_dir) / analysis_name / datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
results_dir.mkdir(parents=True, exist_ok=True)

print("Directory '% s' created" % results_dir)

dat = pd.read_csv(input_file, sep = "\t")
if renamer: dat = dat.rename(columns = renamer) 

if not multiday: 
    dat = dat[dat.week==3] 

# for reproducibility
pyro.set_rng_seed(101)

nt_data = seabass.ScreenData.from_pandas(dat[dat["type"]=="NT"] , device = device) 

essential_data = seabass.ScreenData.from_pandas(dat[dat["type"]=="essential"] , device = device) 


for k,v in nt_posterior.items():
    if np.size(v)==1: print(k,v)

#posterior_stats = seabass.get_posterior_stats(model, guide, data) # can alternatively use guide.median()

plt.hist(nt_posterior["sigma_noise_std"],30)
plt.axvline(np.exp(nt_posterior["log_sigma_noise_mean"]), color="r")
plt.axvline(np.exp( nt_posterior["log_sigma_noise_mean"] + nt_posterior["log_sigma_noise_std"]), color="r")
plt.axvline(np.exp( nt_posterior["log_sigma_noise_mean"] - nt_posterior["log_sigma_noise_std"]), color="r")
plt.show()
# could plot lognormal distribution


plt.hist(nt_posterior["random_slope"],30)
plt.axvline(nt_posterior["slope_noise"], color="r")
plt.axvline(-nt_posterior["slope_noise"], color="r")
plt.show()

from scipy.stats import norm
import statsmodels.stats.multitest # .fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[source]

import lmm_hier 

## figure out noise model on NT guides
# noise_t then guide score prior t
# inf = Gaussian, nan = learnt t, 0 = Laplacian, 1 = Cauchy

# I don't trust comparing ELBO between models with and without hierarchical noise
# because the number of latent variables changes a lot
# Comparing the different settings for noise_t_df and slope_t_df is probably ok though

hierarchical_noise = False
hierarchical_slope = False

elbos = {}

iterations = 3000

for noise_t_df in [0.,1.,5.,np.inf,None]:
    
    for slope_noise in [0.,None]:

        model, guide, losses, nt_posterior = lmm_hier.fit(
            nt_data, 
            hierarchical_noise=hierarchical_noise, 
            hierarchical_slope=hierarchical_slope,
            per_gene_variance = False,
            NT_model=True, 
            iterations=iterations,
            noise_t_df = noise_t_df, 
            slope_noise = slope_noise
        )
        
        elbos[(noise_t_df,slope_noise)] = losses[-1]
        
elbos

elbos_df = pd.DataFrame(
    [ list(k)+[v] for k,v in elbos.items() ],
    columns = ["noise_t_df","slope_noise","neg_elbo"])
elbos_df.noise_t_df = elbos_df.noise_t_df.map('{:.0f}'.format)

elbos_df["random_slope"] = np.isnan(elbos_df.slope_noise)

# slope_noise much better. Learnt 
p9.ggplot(elbos_df[elbos_df.random_slope], p9.aes("noise_t_df", "neg_elbo")) + p9.geom_col(position = "dodge") + p9.coord_cartesian(ylim=[5000,6500])


neg_elbos = {}

iterations = 3000

for noise_t_df in [0.,1.,5.,np.inf,None]:
    
    for slope_t_df in [0.,1.,5.,np.inf,None]:

        model, guide, losses, nt_posterior = lmm_hier.fit(
            nt_data, 
            hierarchical_noise=hierarchical_noise, 
            hierarchical_slope=hierarchical_slope,
            per_gene_variance = False,
            NT_model=True, 
            iterations=iterations,
            noise_t_df = noise_t_df, 
            slope_t_df = slope_t_df, 
            slope_noise = None
        )
        
        neg_elbos[(noise_t_df,slope_t_df)] = losses[-1]
        
neg_elbos

elbos_df = pd.DataFrame(
    [ list(k)+[v] for k,v in neg_elbos.items() ],
    columns = ["noise_t_df","slope_t_df","neg_elbo"])
elbos_df.noise_t_df = elbos_df.noise_t_df.map('{:.0f}'.format)
elbos_df.slope_t_df = elbos_df.slope_t_df.map('{:.0f}'.format)

elbos_df.sort_values("neg_elbo")

p9.ggplot(elbos_df, p9.aes("noise_t_df",  "neg_elbo", fill = "slope_t_df")) + p9.geom_col(position = "dodge") + p9.coord_cartesian(ylim=[5000,6500])

## Run on targetting guides

iterations = 3000
reload(lmm_hier)
per_gene_variance = True
noise_t_df = None
slope_t_df = None
slope_noise = None

all_hypers = ["sigma_noise", "log_slope_noise_mean", "log_slope_noise_std", 'slope_noise', "log_sigma_noise_mean", "log_sigma_noise_std", "noise_t_df", "slope_t_df"]

# assessment: positive slopes are false positives, negative slopes are good.
def get_perf(post, alpha = 0.05):
    z = post["guide_score"] / post["guide_score_se"]
    p = norm.cdf(-np.abs(z))
    sig,q = statsmodels.stats.multitest.fdrcorrection(p, alpha=alpha)
    fp = np.logical_and(z>0, sig)
    tp = np.logical_and(z<0, sig)
    return(np.mean(sig), np.mean(fp), np.mean(tp))

res = {}

for hierarchical_noise in (False, True):
    for hierarchical_slope in (False, True):
        
        try: 
            for i in range(10):
                try: 
                    model, guide, losses, nt_posterior, optim_record = lmm_hier.fit(
                        nt_data, 
                        hierarchical_noise=hierarchical_noise, 
                        hierarchical_slope=hierarchical_slope,
                        per_gene_variance = False, # doesn't make sense for NT_model, there are no genes
                        NT_model=True, 
                        iterations=iterations,
                        noise_t_df = noise_t_df, 
                        slope_noise = slope_noise
                    )
                    break
                except: 
                    raise RuntimeError("failed after 10 attempts")
        except: 
            continue
            
        for t_df in [0.,1.,np.inf,None]:

            try: 
                model, guide, losses, essential_posterior, optim_record = lmm_hier.fit(
                    essential_data, 
                    hierarchical_noise=hierarchical_noise, 
                    hierarchical_slope=hierarchical_slope,
                    per_gene_variance = per_gene_variance,
                    NT_model = False, 
                    iterations=iterations,
                    t_df = t_df,
                    noise_t_df = noise_t_df
                )
            except: 
                continue

            res[(hierarchical_noise,hierarchical_slope,False,t_df)] = get_perf(essential_posterior)

            kwargs = { k : nt_posterior[k].item() for k in nt_posterior if k in all_hypers}
            
            try: 
                model, guide, losses, essential_posterior, optim_record = lmm_hier.fit(
                    essential_data, 
                    hierarchical_noise=hierarchical_noise, 
                    hierarchical_slope=hierarchical_slope,
                    per_gene_variance = per_gene_variance,
                    NT_model = False, 
                    iterations=iterations,
                    t_df = t_df,
                    noise_t_df = noise_t_df, 
                    **kwargs
                )
            except: 
                continue
            
            res[(hierarchical_noise,hierarchical_slope,True,t_df)] = get_perf(essential_posterior)            

with open("lmm_fixed_init.pkl", "wb") as f:
    pickle.dump(res, f)

res_rand_init = pickle.load(open("lmm_fixed_init.pkl", "rb"))

res_rand_init_df = pd.DataFrame([ list(k)+list(v) for k,v in res_rand_init.items() ], columns = ["hier_noise","hier_slope","NT_fit","t","prop_sig","fpr","tpr"])
res_rand_init_df["ttpr"] = res_rand_init_df.tpr - res_rand_init_df.fpr
res_rand_init_df.sort_values("ttpr")

res_rand_init_df[res_rand_init_df.hier_noise & ~res_rand_init_df.hier_slope & res_rand_init_df.NT_fit].sort_values("ttpr")

for k,v in essential_posterior.items():
    if np.size(v)==1: print(k,v)

plt.plot(losses); plt.show()

plt.hist(essential_posterior["sigma_noise_std"],30)
sigma_noise_mean = np.exp(nt_posterior["log_sigma_noise_mean"])
plt.axvline(sigma_noise_mean, color="r")
plt.axvline(np.exp( nt_posterior["log_sigma_noise_mean"] + nt_posterior["log_sigma_noise_std"]), color="r")
plt.axvline(np.exp( nt_posterior["log_sigma_noise_mean"] - nt_posterior["log_sigma_noise_std"]), color="r")
plt.show()

plt.hist(essential_posterior["guide_score"],30)

plt.hist(essential_posterior['guide_score_se'],30)

# uing Laplace noise gives much bigger spread of guide_score_se
# try turning off slope noise: done, just set slope_noise==0. 
# better initialization: quite a bit of work


# Hierarchical noise, not hierarchical for slope noise, and using NT seems best. 
# For t: Gaussian > Laplace > Student-T > Cauchy
hierarchical_noise = True
hierarchical_slope = False

model, guide, losses, nt_posterior, optim_record = lmm_hier.fit(
    nt_data, 
    hierarchical_noise=True, 
    hierarchical_slope=False,
    per_gene_variance = False, # doesn't make sense for NT_model, there are no genes
    NT_model=True, 
    iterations=iterations,
    noise_t_df = None, 
    slope_t_df = None, 
    slope_noise = None
)
plt.hist(nt_posterior["sigma_noise_std"]) # spread of noise variance is pretty small. 0.3-0.4. 

kwargs = { k : nt_posterior[k].item() for k in nt_posterior if k in all_hypers}

res = {}
posts = {}
neg_elbos = {}

for rep in [0,1]: 

    res[rep] = {}
    posts[rep] = {}
    neg_elbos[rep] = {}

    for t_df in [0.,np.inf,None]:

        try: 
            model, guide, losses, posts[rep][t_df], optim_record = lmm_hier.fit(
                essential_data, 
                hierarchical_noise=hierarchical_noise, 
                hierarchical_slope=hierarchical_slope,
                per_gene_variance = per_gene_variance,
                NT_model = False, 
                iterations=iterations,
                t_df = t_df,
                **kwargs
            )

            neg_elbos[rep][t_df] = losses[-1]

            res[rep][t_df] = get_perf(posts[rep][t_df])
            
        except: 
            print("Warning: failed to converge with t_df =", t_df)
            continue

neg_elbos

res

# By ELBO Student-t > Laplacian > Gaussian
res_df = pd.DataFrame([ [k] + [kk] + list(vv) + [neg_elbos[k][kk] ] for k,v in res.items() for kk,vv in v.items()  ], columns = ["rep","t_df","prop_sig","fpr","tpr","neg_elbo"])

res_df["ttpr"] = res_df.tpr - res_df.fpr
res_df.sort_values("ttpr")

essential_posterior = posts[0.0]

plt.scatter(posts[0][0.0]["guide_std"],posts[0][None]["guide_std"])

plt.scatter(posts[0][np.inf]["guide_std"],posts[1][np.inf]["guide_std"])

p9.qplot(posts[0][np.inf]["z"],posts[1][np.inf]["z"], color = posts[1][np.inf]["sig"] + posts[0][np.inf]["sig"]) + p9.xlab("Gaussian prior rep 1") + p9.ylab("Gaussian prior rep 2") + p9.geom_abline(intercept=0, slope=1)


# Student-t vs Laplacian is very similar
# 
p9.qplot(posts[0][0.0]["z"],posts[0][np.inf]["z"], color = posts[0][np.inf]["sig"] + posts[0][0.0]["sig"]) + p9.xlab("Laplacian prior") + p9.ylab("Gaussian prior") + p9.geom_abline(intercept=0, slope=1)

posts[np.inf]["sig"].mean()
posts[0.0]["sig"].mean()

weird_junc = (posts[np.inf]["z"] - posts[0.0]["z"]).argmax()
posts[np.inf]["z"][weird_junc] # -2.2
posts[0.0]["z"][weird_junc] # -9.4
weird_junc_name = essential_data.sgrnas[weird_junc]
p9.ggplot(dat[dat.sgrna == weird_junc_name], p9.aes("week","logFC",color="replicate")) + p9.geom_point() + p9.geom_line()

temp = posts[np.inf]["z"]
temp[ posts[0.0]["sig"] ] = 0.
weird_junc = temp.argmin()
posts[np.inf]["z"][weird_junc] # -2.2
posts[0.0]["z"][weird_junc] # -9.4
weird_junc_name = essential_data.sgrnas[weird_junc]
p9.ggplot(dat[dat.sgrna == weird_junc_name], p9.aes("week","logFC",color="replicate")) + p9.geom_point() + p9.geom_line()

plt.hist(essential_posterior["guide_std"],30)

plt.hist(essential_posterior["guide_score_se"],30)

guide_gene_std = essential_posterior["guide_std"][essential_data.guide_to_gene]

guide_df = pd.DataFrame({
    "guide_score" : essential_posterior["guide_score"], 
    "guide_score_se" : essential_posterior["guide_score_se"],
    "guide_gene_std" : guide_gene_std, 
    "norm_guide_score" : essential_posterior["guide_score"] / guide_gene_std, 
    "norm_guide_score_se" : essential_posterior["guide_score_se"] / guide_gene_std,
    "z" : essential_posterior["guide_score"] / essential_posterior["guide_score_se"]
})

p9.ggplot(guide_df, p9.aes("guide_gene_std", "z", color = "sig0p05")) + p9.geom_point(alpha=0.2)

plt.scatter(guide_gene_std, normalized_guide_score_se, alpha = 0.3)
plt.xlabel("Gene standard deviation")
plt.ylabel("Guide standard error")
plt.show()

plt.hist(essential_posterior["guide_score"][guide_df.sig0p05], 30)

model, guide, losses, essential_posterior, optim_record = lmm_hier.fit(
    essential_data, 
    hierarchical_noise=hierarchical_noise, 
    hierarchical_slope=hierarchical_slope,
    per_gene_variance = per_gene_variance,
    NT_model = False, 
    iterations=iterations,
    t_df = t_df,
    structured_guide = True,
    **kwargs
)

get_perf(essential_posterior)
res_new_df

worst_guide = essential_data.sgrnas[guide_df.guide_score.argmax()]

p9.ggplot(dat[dat.sgrna == worst_guide], p9.aes("week", "logFC")) + p9.geom_point()

dat["rep_week"] = dat["replicate"] + "_" + dat["week"].astype(str)

x = np.array([1,2,3,1,2,3])

dat_wide_essential = pd.pivot(dat[dat.type=="essential"], columns = "rep_week", values = "logFC", index = "sgrna")
slopes_essential = (dat_wide_essential.to_numpy() @ x) / (x.T @ x)
plt.hist(slopes_essential,30)

dat_wide_NT = pd.pivot(dat[dat.type=="NT"], columns = "rep_week", values = "logFC", index = "sgrna")
slopes_NT = (dat_wide_NT.to_numpy() @ x) / (x.T @ x)
plt.hist(slopes_NT,30)

from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(slopes_essential)
plt.plot(ecdf.x,ecdf.y)
ecdf = ECDF(slopes_NT)
plt.plot(ecdf.x,ecdf.y)



(essential_data.sgrnas == dat_wide_essential.index).all()

# Gaussian prior pulls some that probably have inconsistent logFC towards 0
plt.scatter(slopes_essential, posts[0][np.inf]["guide_score"], alpha=0.1) 
plt.scatter(slopes_essential, posts[0][None]["guide_score"], alpha=0.1) # 
plt.scatter(slopes_essential, posts[0][0.0]["guide_score"], alpha=0.1)


plt.scatter(slopes_essential, posts[0][np.inf]["random_slope"][:,0], alpha=0.1) 


x1 = np.array([1,2,3,0,0,0])
random_slope_1 = (dat_wide_essential.to_numpy() @ x1) / (x1.T @ x1)
plt.scatter(random_slope_1, posts[0][np.inf]["random_slope"][:,0], alpha=0.1) 

x2 = np.array([0,0,0,1,2,3])
random_slope_2 = (dat_wide_essential.to_numpy() @ x2) / (x2.T @ x2)
plt.scatter(random_slope_2, posts[0][np.inf]["random_slope"][:,1], alpha=0.1) 

fitted = np.outer(slopes_essential, x)
resid = dat_wide_essential.to_numpy() - fitted

random_slope_1 = (resid @ x1) / (x1.T @ x1)
plt.scatter(random_slope_1, posts[0][np.inf]["random_slope"][:,0], alpha=0.1) 

random_slope_2 = (resid @ x2) / (x2.T @ x2)
plt.scatter(random_slope_2, posts[0][np.inf]["random_slope"][:,1], alpha=0.1) 

neg_elbos = {}
res = {}
posts = {}

for init_gs in [True,False]: 
    for init_rs in [True,False]: 

        model, guide, losses, posts[(init_gs,init_rs)], optim_record = lmm_hier.fit(
            essential_data, 
            hierarchical_noise=hierarchical_noise, 
            hierarchical_slope=hierarchical_slope,
            per_gene_variance = per_gene_variance,
            NT_model = False, 
            iterations=iterations,
            t_df = t_df,
            structured_guide = True,
            init_random_slopes = np.vstack([random_slope_1,random_slope_2]).T if init_rs else None, 
            init_guide_scores = slopes_essential if init_gs else None,
            **kwargs
        )
        
        neg_elbos[(init_gs,init_rs)] = losses
        
        res[(init_gs,init_rs)] = get_perf(posts[(init_gs,init_rs)])

neg_elbos

for init_gs in [True,False]: 
    for init_rs in [True,False]: 
        plt.plot(neg_elbos[(init_gs,init_rs)], label = "init guide score %i init random slope %i" % (init_gs,init_rs))
plt.legend()

plt.scatter(posts[(True,True)]["z"], posts[(False,False)]["z"], alpha = 0.1)
plt.scatter(posts[(True,False)]["z"], posts[(False,False)]["z"], alpha = 0.1)
plt.scatter(posts[(False,True)]["z"], posts[(False,False)]["z"], alpha = 0.1)

# All agree pretty well
{k:np.mean(v[-500:]) for k,v in neg_elbos.items()} # all basically get to the same solution



iterations = 1000

neg_elbos = {}
res = {}
posts = {}

for rep in range(2):

    for finetune_iterations in [100,200]: 
        
        k = (rep,finetune_iterations)
        
        model, guide, losses, posts[k], optim_record = lmm_hier.fit(
            essential_data, 
            hierarchical_noise=hierarchical_noise, 
            hierarchical_slope=hierarchical_slope,
            per_gene_variance = per_gene_variance,
            NT_model = False, 
            iterations=iterations + (0 if finetune_iterations else 900), 
            t_df = t_df,
            structured_guide = True,
            finetune_iterations = finetune_iterations, 
            **kwargs
        )
        
        neg_elbos[k] = losses
        
        res[k] = get_perf(posts[k])

plt.plot(neg_elbos[(0,200)])

plt.scatter(posts[(0,200)]["z"], posts[(1,200)]["z"])

plt.scatter(posts[(0,0)]["z"], posts[(1,0)]["z"])

losses = neg_elbos[(0,200)]

window = 10

R,p = zip(*[ scipy.stats.pearsonr(np.arange(window), losses[i-window:i]) for i in np.arange(window, len(losses)) ])

plt.plot(R)
plt.plot(-np.log10(p))

plt.plot(losses)
