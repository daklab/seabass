# Screen Efficacy Analysis with BAyesian StatisticS

SEABASS is a hierarchical linear mixed model for analysing CRISPR screen data. It can handle multiple time-points and replicates. It uses stochastic variational inference, implemented in `pyro` to fit model parameters. This enables using heavy-tailed noise distributions which provide a better fit to data and robustness to outliers. 

## Probabilistic model

The probabilistic model for SEABASS is: 

* guide_score ~ Normal(0, guide_std^2) for each guide
* log2FC = (guide_score + guide_random_slope) * timepoint + noise
* noise ~ D1(0, sigma_noise) for each observation
* guide_random_slope ~ D2(0, slope_noise) for each (guide,replicate) pair

where guide_score is a slope and D1 and D2 are location-scale distributions which can be either normal, Cauchy, Laplace or StudentT. 

The noise standard deviation (std) can either be shared across guides (hierarchical_noise = False), or per guide but distributed according to a learned prior (hierarchical_noise = True): 

noise_std ~ logNormal(log_guide_std_mean,log_guide_std_std^2)

Similarly slope_noise can either be shared shared guides (hierarchical_slope = False), or per guide but distributed according to a learned prior (hierarchical_noise = True):

slope_noise ~ logNormal(log_sigma_noise_mean,log_sigma_noise_std^2)

Additionally SEABASS can learn a per gene guide_std ~ logNormal(log_guide_std_mean, log_guide_std_std^2) to account for differences in essentiality. 

## Installation

`pip install seabass`

## Usage

See `example_usage/example.ipynb`