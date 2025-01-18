library(torch)
library(lavaan)
library(psych)
library(sdpt3r)
library(fungible)
library(misty)
library(MASS)  

source("approaches.R")
source("ModelDefine.R")
numofItems <- 12
n_factors <- 2
df <- 53
# Define lambda structure as a binary matrix indicating fixed vs. free parameters
lambda_structure <- matrix(c(
  1, 0,
  1, 0,
  1, 0,
  1, 0,
  1, 0,
  1, 0,
  0, 1,
  0, 1,
  0, 1,
  0, 1,
  0, 1,
  0, 1
), nrow = 12, byrow = TRUE)
lambda_mask <- torch_tensor(lambda_structure)
for(size in c(100, 200,500))
{
  for(loading in c(3,6,9))
  {
    nameofmodel <- get(paste0("pop.mod",".", loading, sep = ""))
    mod_constraint <- get("mod_constraint")
    cat(nameofmodel, "\n")
    se_1 <- c()
    se_2 <- c()
    se_3 <- c()
    se_4 <- c()
    se_5 <- c()
    upper_1 <- c()
    upper_2 <- c()
    upper_3 <- c()
    upper_4 <- c()
    upper_5 <- c()
    lower_1 <- c()
    lower_2 <- c()
    lower_3 <- c()
    lower_4 <- c()
    lower_5 <- c()
    result_loading_1 <- c()
    result_loading_2 <- c()
    result_loading_3 <- c()
    result_loading_4 <- c()
    result_loading_5 <- c()
    heywood <- 0
    for (i in c(1:500)) {
    cat("loading:",loading/10,";size:", size,";i:",i, "\n")
    set.seed(i+100)
    datb <- simulateData(nameofmodel, sample.nobs=size)-1
    fit_method1 <- lavaan::cfa(mod_no_constraint, data=datb)
    fit_method2 <- lavaan::cfa(mod_constraint, data=datb)
    result_lavaan_1 <- summary(fit_method1, fit.measures=TRUE, rsquare=TRUE, standardized=TRUE)
    ci_95_upper <- result_lavaan_1$pe$est[1:numofItems] + 1.96*result_lavaan_1$pe$se[1:12]
    upper_1 <- rbind(upper_1, ci_95_upper)
    ci_95_lower <- result_lavaan_1$pe$est[1:numofItems] - 1.96*result_lavaan_1$pe$se[1:12]
    lower_1 <- rbind(lower_1, ci_95_lower)
    
    result_lavaan_2 <- summary(fit_method2, fit.measures=TRUE, rsquare=TRUE, standardized=TRUE)
    
    #rmsea_lavaan_1 <- rbind(rmsea_lavaan,result_lavaan$fit["rmsea"] )
    result_loading_1 <<- rbind(result_loading_1, result_lavaan_1$pe$est[1:numofItems])
    result_loading_2 <<- rbind(result_loading_2, result_lavaan_2$pe$est[1:numofItems])
       
    ci_95_upper <- result_lavaan_2$pe$est[1:numofItems] + 1.96*result_lavaan_2$pe$se[1:12]
    upper_2 <- rbind(upper_2, ci_95_upper)
    
    ci_95_lower <- result_lavaan_2$pe$est[1:numofItems] - 1.96*result_lavaan_2$pe$se[1:12]
    lower_2 <- rbind(lower_2, ci_95_lower)
    if(any(result_lavaan_1$pe$est[(numofItems+4):(numofItems+3+12)]<0))
    {
      heywood <- heywood + 1
      print(result_lavaan_1$pe$est[(numofItems+4):(numofItems+3+12)])
    }
    try({
       result3 <- mle_method(datb, 3, n = size)
       result4 <- mle_method(datb, 4, n = size)
       result5 <- mle_method(datb, 5, n = size)
    },
    next )
    
    result_loading_3 <- rbind(result_loading_3,rowSums(as.matrix(result3$lambda*lambda_mask)))
    result_loading_4 <- rbind(result_loading_4,rowSums(as.matrix(result4$lambda*lambda_mask)))
    result_loading_5 <- rbind(result_loading_5,rowSums(as.matrix(result5$lambda*lambda_mask)))
    
    upper_3 <- rbind(upper_3,result3$upper)
    upper_4 <- rbind(upper_4,result4$upper)
    upper_5 <- rbind(upper_5,result5$upper)
    lower_3 <- rbind(lower_3,result3$lower)
    lower_4 <- rbind(lower_4,result4$lower)
    lower_5 <- rbind(lower_5,result5$lower)
    }
    true_loading <- loading/10
    ###### find the percentage of true loading inside the 95% confidence interval ------------
    ci_95 <- rbind(
    percentage95(upper_1, lower_1, true_loading),
    percentage95(upper_2, lower_2, true_loading),
    percentage95(upper_3, lower_3, true_loading),
    percentage95(upper_4, lower_4, true_loading),
    percentage95(upper_5, lower_5, true_loading)
    )
    rmse_result <- rbind(
      rmse(c(result_loading_1), true_loading),
      rmse(c(result_loading_2), true_loading),
      rmse(c(result_loading_3), true_loading),
      rmse(c(result_loading_4), true_loading),
      rmse(c(result_loading_5), true_loading)
    )
    result <- cbind(ci_95,rmse_result, heywood)
    write.csv(result, paste0(loading,"-size",size,"-result",".csv"))
    
    loading1 <- rmse_condition(result_loading_1, true_loading)
    loading2 <- rmse_condition(result_loading_2, true_loading)
    loading3 <- rmse_condition(result_loading_3, true_loading)
    loading4 <- rmse_condition(result_loading_4, true_loading)
    loading5 <- rmse_condition(result_loading_5, true_loading)
    approach=c(rep("Method1",length(loading1)),rep("Method2",length(loading2)),rep("Method3",length(loading3)),rep("Method4",length(loading4)),rep("Method5",length(loading5)))
    data_agg=data.frame(value=c(c(loading1),c(loading2),c(loading3),c(loading4),c(loading5)),  approach = approach)
    write.csv(data_agg, paste0(loading,"-size",size,"-loading",".csv"))
  }
}


