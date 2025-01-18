
############# initialize -----
inverse_sigmoid <- function(x)
{
  -log(2/(x + 1) - 1)
}
# CFA Log-likelihood function with improved stability
cfa_log_likelihood_map_exp <- function(lambda, theta_delta, phi, sample_cov, n, lambda_mask) {
  # Ensure theta_delta is diagonal
  theta_delta_diag <- torch_diag(exp(theta_delta)) 
  phi_diag <- phi.matrix(phi)$requires_grad_(T)
  lambda <- (lambda)*lambda_mask

  implied_cov <- torch_matmul(torch_matmul(lambda, phi_diag),(lambda$t())) + theta_delta_diag
  # Log determinants
  logdet_implied <- torch_logdet(implied_cov)
  logdet_sample <- torch_logdet(sample_cov)
  n_variables <- nrow(sample_cov)
  # Inverse of implied covariance matrix
  inv_implied <- torch_inverse(implied_cov )
  penalty_weight <- 1
  log_likelihood <- -0.5 *n* (n_variables*log(2*pi) + logdet_implied + sum(torch_diag(torch_matmul(sample_cov, inv_implied)))) - torch_tensor((lambda^2)/(0.2^2))$sum()           
  return(-log_likelihood)  # Negative log-likelihood for minimization
}

cfa_log_likelihood_map_ll <- function(lambda, theta_delta, phi, sample_cov, n, lambda_mask) {
  # Ensure theta_delta is diagonal
  theta_delta_diag <- torch_diag(exp(theta_delta)) 
  phi_diag <- phi.matrix(phi)$requires_grad_(T)
  lambda <- (lambda)*lambda_mask
  
  implied_cov <- torch_matmul(torch_matmul(lambda, phi_diag),(lambda$t())) + theta_delta_diag
  # Log determinants
  logdet_implied <- torch_logdet(implied_cov)
  logdet_sample <- torch_logdet(sample_cov)
  n_variables <- nrow(sample_cov)
  # Inverse of implied covariance matrix
  inv_implied <- torch_inverse(implied_cov )
  penalty_weight <- 1
  log_likelihood <- -0.5 *n* (n_variables*log(2*pi) + logdet_implied + sum(torch_diag(torch_matmul(sample_cov, inv_implied))))
  #cat(as.numeric(inv_implied[1,1]), ",", as.numeric(log_likelihood), "\n")
  return(-log_likelihood)  # Negative log-likelihood for minimization
}

cfa_log_likelihood_sigmoid_exp <- function(lambda, theta_delta, phi, sample_cov, n, lambda_mask) {
  # Ensure theta_delta is diagonal
  theta_delta_diag <- torch_diag(exp(theta_delta)) 
  phi_diag <- phi.matrix(phi)$requires_grad_(T)
  lambda <- sigmoid(lambda)*lambda_mask
  
  implied_cov <- torch_matmul(torch_matmul(lambda, phi_diag),(lambda$t())) + theta_delta_diag
  regularization_constant<- 1e-20
  implied_cov <- implied_cov + diag(regularization_constant, nrow(implied_cov))  # Regularize to avoid singularity
  n_variables <- nrow(sample_cov)
  # Log determinants
  logdet_implied <- torch_logdet(implied_cov)
  logdet_sample <- torch_logdet(sample_cov)
  
  # Inverse of implied covariance matrix
  inv_implied <- torch_inverse(implied_cov )
  log_likelihood <- -0.5 *n* (n_variables*log(2*pi) + logdet_implied + sum(torch_diag(torch_matmul(sample_cov, inv_implied))))           
  #cat(as.numeric(inv_implied[1,1]), ",", as.numeric(log_likelihood), "\n")
  return(-log_likelihood)  # Negative log-likelihood for minimization
}


cfa_log_likelihood_exp <- function(lambda, theta_delta, phi, sample_cov, n, lambda_mask) {
  # Ensure theta_delta is diagonal
  theta_delta_diag <- torch_diag(exp(theta_delta)) 
  lambda <- lambda*lambda_mask
  
  phi_diag <- phi.matrix(phi)$requires_grad_(T) 
  implied_cov <- torch_matmul(torch_matmul(lambda, phi_diag),(lambda$t())) + theta_delta_diag
  regularization_constant<- 1e-20
  implied_cov <- implied_cov + diag(regularization_constant, nrow(implied_cov))  # Regularize to avoid singularity
  
  # Log determinants
  logdet_implied <- torch_logdet(implied_cov)
  logdet_sample <- torch_logdet(sample_cov)
  n_variables <- nrow(sample_cov)
  # Inverse of implied covariance matrix
  inv_implied <- torch_inverse(implied_cov )
  penalty_weight <- 1
  log_likelihood <- -0.5 *n* (n_variables*log(2*pi) + logdet_implied + sum(torch_diag(torch_matmul(sample_cov, inv_implied))))           
  #cat(as.numeric(inv_implied[1,1]), ",", as.numeric(log_likelihood), "\n")
  return(-log_likelihood)  # Negative log-likelihood for minimization
}

sigmoid <- function(x)
{
  -1 + 2/(1+exp(-x))
 
}
exp_input <- function(x)
{
  exp(x)
}
delta_method <- function(x){
  2*exp(-x)/(1+exp(-x))^2
}
vec <- function(x) {
  torch_cat(lapply(x, function(x) {
    x$view(-1)
  }))
}
bic <- function(logl, npar, N)
{
  -2*logl + npar*log(N)
}

phi.matrix <- function(v)
{
  K <- length(v)
  if(K == 1) K = 2
  z <- v$tanh()
  Sigma1.L.mask <- torch_tensor(lower.tri(matrix(0, K, K)))$bool()
  Sigma1.L <- torch_eye(K)$masked_scatter(Sigma1.L.mask, z) * torch_cat(c(torch_ones(K, 1), (1 - torch_zeros(K, K)$masked_scatter(Sigma1.L.mask, z)[, 1:-2]$square())$cumprod(2)$sqrt()), 2)
  torch_matmul(Sigma1.L,  Sigma1.L$t())
}
percentage95 <- function(upper, lower, true){
  i <- 1
  numofCI <- 0
  while (i<length(upper)+1) {
    
    if(!is.na(upper[i]) && !is.na(lower[i]) && as.numeric(upper[i]) > true && as.numeric(lower[i]) < true)
    {
      numofCI = numofCI + 1
    }
    i <- i + 1
  }
  numofCI/length(!is.na(upper))
}
rmse <- function(estimation, true)
{
  len = length(estimation)
  bias = 0
  for(i in c(1:len))
  {
    bias = bias + (estimation[i] - true)^2 
  }
  sqrt(bias/len)
}
rmse_condition <- function(loading, true)
{
  loading_rmse <- c()
  len <- nrow(loading)
  for(i in c(1:len))
  {
    #print(loading[i,])
    temp <- rmse(loading[i,],true)
    loading_rmse <- c(loading_rmse,temp )
  }
  loading_rmse
}

mle_method <- function(datb, method = 3, tolerance = 1e-3, n, str){
# Load your sample covariance data (assume 'datb' is loaded)
sample_cov <- cov(datb)  # Base covariance matrix from your data
sample_cov_tensor <- torch_tensor(sample_cov)  # Convert to torch tensor
# Number of observed variables and latent factors
n_variables <- nrow(sample_cov)
n_factors <- ncol(str)
result <- c()
# Initialize lambda with specific values at non-zero positions defined by lambda_structure
initial_lambda_values <- matrix(0, nrow = n_variables, ncol = n_factors)
 # Set non-zero loadings to an initial value
# Initializing theta_delta with the diagonal of sample covariance
  if(method == 3) {
  initial_lambda_values[str == 1] <- 0.1 
  cfa_function <- cfa_log_likelihood_exp
  }else if(method == 4) {
    initial_lambda_values[str == 1] <- inverse_sigmoid(0.1)
    cfa_function <- cfa_log_likelihood_sigmoid_exp
  }else if(method == 5) {
    initial_lambda_values[str == 1] <- 0.1
    cfa_function <- cfa_log_likelihood_map_exp
  }else if(method == 0) {
    initial_lambda_values[str == 1] <- 0.1
    cfa_function <- cfa_log_likelihood_test
  }
  
lambda <- torch_tensor(initial_lambda_values, requires_grad = TRUE)
theta_delta <- torch_tensor(log(diag(sample_cov)), requires_grad = TRUE)
if(method == 0) theta_delta <- torch_tensor(diag(sample_cov), requires_grad = TRUE)
val <- rep(0.3, (n_factors*(n_factors -1)/2))
phi <- torch_tensor(val, requires_grad = TRUE)
# Adam optimizer with a smaller learning rate for stability
optimizer <- optim_adam(params = list(lambda, theta_delta, phi), lr = 0.1)

# Optimization loop with more epochs
num_epochs <- 100000

last_loss <- NULL
lambda_mask <- torch_tensor(str)
for (epoch in 1:num_epochs) {
  optimizer$zero_grad()  # Zero out gradients

  loss <- cfa_function(lambda, theta_delta, phi, sample_cov_tensor, n, lambda_mask)  # Compute loss
  loss$backward()  # Backpropagation
  
  # Apply gradient mask to enforce structure, keeping gradients only where lambda_structure == 1
  with_no_grad({
    lambda$grad <- lambda$grad * lambda_mask  # Zero out gradients where lambda_structure == 0
  })
  
  optimizer$step()  # Update parameters
  
  # Early stopping condition
  if (!is.null(last_loss) && abs(last_loss - loss$item()) < tolerance) {
    cat("Early stopping at epoch:", epoch, "with loss:", loss$item(), "\n")
    loss <- cfa_function(lambda, theta_delta, phi, sample_cov_tensor, n, lambda_mask)  # Compute loss
    loss$backward(retain_graph = TRUE)
    
    # # Step 1: Compute first-order gradients with autograd_grad
    # first_derivative_lambda <- autograd_grad(loss, inputs = list(lambda, theta_delta, phi), create_graph = TRUE)
    # 
    # result <- lapply(vec(first_derivative_lambda)$unbind(),function(l) {
    #   vec(autograd_grad(l, list(lambda, theta_delta, phi), retain_graph = T))
    # })
    # var_x <- c(as.array(torch_diag(torch_stack(result)$pinverse()))[1:(2*n_variables)])
    # var <- c(var_x[c(1,3,5,7,9,11,14,16,18,20,22,24)])
    # ci_95_upper <- (lambda*lambda_structure)$sum(2) + 1.96*sqrt(var)
    # ci_95_lower <-  (lambda*lambda_structure)$sum(2) - 1.96*sqrt(var)
    if(method == 4) 
    {
    lambda <- sigmoid(lambda)
    #   var <- c(t(as.array(delta_method(lambda)^2)))*var
    #   ci_95_upper <- sigmoid(ci_95_upper)
    #   ci_95_lower <- sigmoid(ci_95_lower)
    #   
    }
    if(method == 5) 
    {
      loss <- cfa_log_likelihood_map_ll(lambda, theta_delta, phi, sample_cov_tensor, n, lambda_mask)  # Compute loss
      loss$backward(retain_graph = TRUE)
      
      #   var <- c(t(as.array(delta_method(lambda)^2)))*var
      #   ci_95_upper <- sigmoid(ci_95_upper)
      #   ci_95_lower <- sigmoid(ci_95_lower)
      #   
    }
    #  se <- sqrt(var)

    if(method == 0) result <- list(lambda = as.matrix(lambda),theta = as.matrix(theta_delta), phi = as.matrix(phi.matrix(phi)), ll = as.numeric(loss$item()))
    
    else result <- list(lambda = as.matrix(lambda),theta = as.matrix(exp(theta_delta)), phi = as.matrix(phi.matrix(phi)), ll = as.numeric(loss$item()))
    result
    break
  }
  
  
  last_loss <- loss$item()
  }
  return(result)
}

library(semfindr)
data("cfa_dat_heywood")
help(data("cfa_dat_heywood"))
# Specify the CFA model
model <- '
  f1 =~ NA*x1 + x2 + x3
  f2 =~ NA*x4 + x5 + x6
  f1 ~~ 1*f1
  f2 ~~ 1*f2
'
lambda_structure <- matrix(c(
  1, 0,
  1, 0, 
  1, 0, 
  0, 1, 
  0, 1, 
  0, 1
), nrow = 6, byrow = TRUE)
# Fit the model
fit <- cfa(model, data = cfa_dat_heywood)
# Display parameter estimates
result <- summary(fit,  fit.measures=TRUE, standardized = TRUE) #bic: 1186.017

model2 <- '
  f1 =~ NA*x1 + x2 + x3
  f2 =~ NA*x4 + x5 + x6
  f1 ~~ 1*f1
  f2 ~~ 1*f2
  x1 ~~ a*x1
  a > 0
'
fit2 <- cfa(model2, data = cfa_dat_heywood)

cfa_dat_heywood <- cfa_dat_heywood[,c(1,2,3,5,4,6)]
# Display parameter estimates
result2 <- summary(fit2,  fit.measures=TRUE, standardized = TRUE) #bic: 1186.017
result2$fit
m3 <- mle_method(cfa_dat_heywood, method = 3, n = 60, str = lambda_structure) #1192.284
bic(-m3$ll,npar=13,N = 60)
m4 <- mle_method(cfa_dat_heywood, method = 4, n = 60, str = lambda_structure) # 1192.48
bic(-m4$ll,npar=13,N = 60)
m5 <- mle_method(cfa_dat_heywood, method = 5, n = 60, str = lambda_structure) #1192.094
bic(-m5$ll,npar=13,N = 60)
