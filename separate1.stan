
data {
  int<lower=0> J; //number of groups
  int<lower=0> N1; //number of observations for group 1
  int<lower=0> N2; //number of observations for group 2
  int<lower=0> N3; //number of observations for group 3
  int<lower=1> K; //number of features
  
  matrix[N1, K] x1;
  matrix[N2, K] x2;
  matrix[N3, K] x3;
  
  vector[N1] y1;
  vector[N2] y2;
  vector[N3] y3;
  
  int prior;
}


parameters {
  real alpha1;
  vector[K] beta1;
  
  real alpha2;
  vector[K] beta2;
  
  real alpha3;
  vector[K] beta3;
  
  real<lower=0> sigma;
}

model {
  // default priors 
  if (prior == 1) {
    alpha1 ~ normal(110, 30);
    beta1 ~ normal(0, 1);
    
    alpha2 ~ normal(110, 30);
    beta2 ~ normal(0, 1);
    
    alpha3 ~ normal(110, 30);
    beta3 ~ normal(0, 1);
  
    sigma ~ normal(0, 10);
  }
  // wide priors
  else if (prior == 2) { 
    alpha1 ~ normal(0, 100);
    beta1 ~ normal(0, 2);
    
    alpha2 ~ normal(0, 100);
    beta2 ~ normal(0, 2);
    
    alpha3 ~ normal(0, 100);
    beta3 ~ normal(0, 2);
  
    sigma ~ normal(0, 30);
    
  }
  // narrow priors
  else if (prior == 3) { 
    alpha1 ~ normal(110, 5);
    beta1 ~ normal(0, 0.5);
    
    alpha2 ~ normal(110, 5);
    beta2 ~ normal(0, 0.5);
    
    alpha3 ~ normal(110, 5);
    beta3 ~ normal(0, 0.5);
  
    sigma ~ normal(0, 5);
  }

  
  //likelihood
  y1 ~ normal(alpha1 + x1*beta1, sigma);
  y2 ~ normal(alpha2 + x2*beta2, sigma);
  y3 ~ normal(alpha3 + x3*beta3, sigma);
}

generated quantities {
  //posterior predictive distribution for posterior predictive check
  real y1_rep[N1] = normal_rng(alpha1 + x1*beta1, sigma);
  real y2_rep[N2] = normal_rng(alpha2 + x2*beta2, sigma);
  real y3_rep[N3] = normal_rng(alpha3 + x3*beta3, sigma);
  
  //log-likelihood
  vector[N1+N2+N3] log_lik;
  
  for (i in 1:N1) {
    log_lik[i] = normal_lpdf(y1[i] | x1[i] * beta1 + alpha1, sigma);
  }
  
  for (i in 1:N2) {
    log_lik[i+N1] = normal_lpdf(y2[i] | x2[i] * beta2 + alpha2, sigma);
  }
  
  for (i in 1:N3) {
    log_lik[i+N1+N2] = normal_lpdf(y3[i] | x3[i] * beta3 + alpha3, sigma);
  }
}

