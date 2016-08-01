rm(list=ls())
library(corrplot); library(ROCR); library(caTools)
## Functions for roc-auc and pr-auc
## Originally written by Andy Beger
auc_roc <- function(obs, pred) {
  pred <- prediction(pred, obs)
  auc  <- performance(pred, "auc")@y.values[[1]]
  return(auc)
}

auc_pr <- function(obs, pred) {
  xx.df <- prediction(pred, obs)
  perf  <- performance(xx.df, "prec", "rec")
  xy    <- data.frame(recall=perf@x.values[[1]], precision=perf@y.values[[1]])
  
  # take out division by 0 for lowest threshold
  xy <- subset(xy, !is.nan(xy$precision))
  
  res   <- trapz(xy$recall, xy$precision)
  res
}

rocdf <- function(pred, obs, data=NULL, type=NULL) {
  # plot_type is "roc" or "pr"
  if (!is.null(data)) {
    pred <- eval(substitute(pred), envir=data)
    obs  <- eval(substitute(obs), envir=data)
  }
  
  rocr_xy <- switch(type, roc=c("tpr", "fpr"), pr=c("prec", "rec"))
  rocr_df <- prediction(pred, obs)
  rocr_pr <- performance(rocr_df, rocr_xy[1], rocr_xy[2])
  xy <- data.frame(rocr_pr@x.values[[1]], rocr_pr@y.values[[1]])
  colnames(xy) <- switch(type, roc=c("tpr", "fpr"), pr=c("rec", "prec"))
  return(xy)
}

load('data/all_preds.rda')
## Dates for validation/test sets
calib.start <- as.Date("2010-01-01")
calib.end   <- as.Date("2012-04-30")
test.start  <- calib.end+1
test.end    <- as.Date("2014-03-31")
## Get Ward test predictions
testpreds <- all.preds[all.preds$date>=test.start & all.preds$date<=test.end, ]
## Read test data
testfull <- read.csv('test.csv')

## Load Ward logit predictions
log.preds <- read.csv('data/logit_pred.csv')
testfull$logit <- log.preds[,2]
## Load preds from other models
otherpreds <- read.csv('data/otherprobs.csv')
testfull <- cbind(testfull, otherpreds[,1:3])
## Load preds from NN
nnpreds <- read.csv('data/predictionsRELU40.txt', header=F)
colnames(nnpreds) <- 'RELU'
testfull <- cbind(testfull, nnpreds)
testfull <- testfull[,c('country', 'date', 'failure', 'logit', 'boosting', 'rf', 'elasticnet', 'RELU')]

names(testfull)
names(testpreds)
testfull$date <- as.Date(testfull$date, format = "%Y-%m-%d")
preds <- merge(testfull,testpreds,by=c("country","date"))

preds <- preds[,c('country', 'date', 'failure','irr.t', 'ch.Ensemble', 'logit', 'boosting', 'rf', 'elasticnet', 'RELU')]
names(preds)[5] <- 'ebma'
auc_pr(obs= preds$failure, pred=preds$ebma)
auc_roc(obs= preds$failure, pred=preds$ebma)
auc_pr(obs= preds$failure, pred=preds$RELU)
auc_roc(obs= preds$failure, pred=preds$RELU)
auc_roc(obs= preds$failure, pred=preds$RELU)

predcorr <- cor(preds[,c(5:10)])
preds[preds$failure==1,]

## ROC AUC plot
pdf('plots/roc_curve.pdf', width=9, height=7)
roc.xgb <- rocdf(preds$boosting, preds$failure, type="roc")
roc.rf <- rocdf(preds$rf, preds$failure, type="roc")
roc.nn <- rocdf(preds$RELU, preds$failure, type="roc")
roc.ebma <- rocdf(preds$ebma, preds$failure, type="roc")
roc.elasticnet <- rocdf(preds$elasticnet, preds$failure, type="roc")
plot(roc.nn[, 1], roc.nn[, 2], type='l', col="#66C2A5", lty=2,lwd=1.5,
     ylab="True Positive Rate",
     xlab="False Positive Rate",
     main='ROC Curve',
     cex.main = 2, cex.lab=1.5, cex.axis = 1.1)
lines(roc.xgb[, 1], roc.xgb[, 2], type='l', col= "#8DA0CB", lty=2, lwd=1.5)
lines(roc.rf[, 1], roc.rf[, 2], type='l', col="#E78AC3", lty=2, lwd=1.5)
lines(roc.elasticnet[, 1], roc.elasticnet[, 2], type='l', col="#A6D854", lty=2, lwd=1.5)
lines(roc.ebma[, 1], roc.ebma[, 2], type='l', col="#FC8D62", lty=2, lwd=1.5)
abline(a = 0, b = 1, lty=2, col='gray30', lwd=0.5)
legend(x='bottomright', c('NN ReLU(40) = 0.868', 'EBMA =0.839',
                          'Gradient Boosting = 0.757', 'Random Forests = 0.751',
                          'Elastic Net Logit=0.829'),
       text.col=c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"),
       bty = "n", cex=1.6)
dev.off()
pdf('plots/pr_curve.pdf', width=9, height=7)
pr.xgb <- rocdf(preds$boosting, preds$failure, type="pr")
pr.rf <- rocdf(preds$rf, preds$failure, type="pr")
pr.nn <- rocdf(preds$RELU, preds$failure, type="pr")
pr.ebma <- rocdf(preds$ebma, preds$failure, type="pr")
pr.elasticnet <- rocdf(preds$elasticnet, preds$failure, type="pr")
plot(pr.nn[, 1], pr.nn[, 2], type='l', col="#66C2A5", lty=2,lwd=1.5,
     ylab="Precision",
     xlab="Recall",
     main='Precision Recall Curve',
     cex.main = 2, cex.lab=1.5, cex.axis = 1.1)
lines(pr.xgb[, 1], pr.xgb[, 2], type='l', col="#8DA0CB", lty=2, lwd=1.5)
lines(pr.rf[, 1], pr.rf[, 2], type='l', col="#E78AC3", lty=2, lwd=1.5)
lines(pr.elasticnet[, 1], pr.elasticnet[, 2], type='l', col= "#A6D854", lty=2, lwd=1.5)
lines(pr.ebma[, 1], pr.ebma[, 2], type='l', col="#FC8D62", lty=2, lwd=1.5)
legend(x='topright', c('NN ReLU(40) = 0.020', 'EBMA = 0.016',
                          'Gradient Boosting = 0.005', 'Random Forests = 0.004',
                          'Elastic Net Logit = 0.016'),
       text.col=c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"),
       bty = "n", cex=1.6)
dev.off()


pdf('plots/predicition_corr.pdf')
corrplot(predcorr, method="pie")
dev.off()