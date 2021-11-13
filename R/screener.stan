data {
  real<lower=0> efficacy_prior_a; // shape1 of beta(a,b) prior on guide efficacy
  real<lower=0> efficacy_prior_b; // shape2 of beta(a,b) prior on guide efficacy
  real sigma_noise; // noise std estimated from non-targetting guides
  real sigma_prior; 
  int<lower=0> N; // number of measurents
  vector[N] y; // log folds changes
  int which_guide[N]; 
  int which_junction[N];
  int<lower=0> num_guides; // number of measurents
  int<lower=0> num_junctions; 
}
parameters {
  real<lower=0, upper=1> guide_efficacy[num_guides];
  real junction_essentiality[num_junctions];
}
model {
  junction_essentiality ~ normal(0, sigma_prior);
  guide_efficacy ~ beta(efficacy_prior_a, efficacy_prior_b);
  for (i in 1:N) {
    y[i] ~ normal(guide_efficacy[which_guide[i]] * junction_essentiality[which_junction[i]], sigma_noise);
  }
}

