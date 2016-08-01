rm(list=ls())
getwd()
library(caret); library(ROCR); library(caTools)

## Functions for roc-auc and pr-auc
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



## Set seed
set.seed(1900)
## Load data
load('data/prepared_data.RData')

## Rbind train and validation sets for caret (manually program validation set in tc)
data <- rbind(train, valid)

## Formula for training in caret
form <- paste0('failure',"~", paste0(colnames(data)[6:37], collapse = "+"))
form

log <- glm(as.formula(form), data=train, family='binomial')
preds <- predict(log, newdata=test)
## Observed outcome in test set for auc calculations
obs <- as.numeric(test$failure)-1
auc_roc(obs, preds)
## Set up data for training in caret
tc <- trainControl(method="cv", 
                number=1, ## one fold (i.e. specified validation set)
                summaryFunction=twoClassSummary, ## provides ROC summary stats in call to model
                classProb=T, ## return predicted probs
                index = list(Fold1 = (1:17384)),
                indexFinal= c(1:17384)) ## train on training set obs

## Hyperparameters for random forest
rfGrid <-  expand.grid(mtry = c(2, 4, 5, 7, 9, 10)) ##number of features to select from at each split
## Run rf in caret
model.rf<-train(as.formula(form), 
                  method="rf", ## random forest model
                  importance=T, 
                  proximity=F,
                  trControl=tc,
                  tuneGrid = rfGrid,
                  data=data)
## Model output
model.rf
## rf predictions
rf.pprob <- predict(model.rf, test, type="prob")
rf.roc.auc <- auc_roc(obs, rf.pprob$change)
rf.pr.auc <- auc_pr(obs, rf.pprob$change)
rf.roc.auc
rf.pr.auc


## Run xgb in caret
model.xgb <-train(as.formula(form), 
                  metric="ROC", method="xgbTree", ## stochastic gradient boosting model
                  trControl=tc,
                  data=data)
## Model output
model.xgb
## xgb predictions
xgb.pprob <- predict(model.xgb, test, type="prob")
xgb.roc.auc <- auc_roc(obs, xgb.pprob$change)
xgb.pr.auc <- auc_pr(obs, xgb.pprob$change)
xgb.roc.auc
xgb.pr.auc

## Run logistic regression in caret
model.logit <-train(as.formula(form), 
                  metric="ROC", method="glm", ## stochastic gradient boosting model
                  trControl=tc,
                  data=data)
## Model output
model.logit
## logit predictions
logit.pprob <- predict(model.logit, test, type="prob")
logit.roc.auc <- auc_roc(obs, logit.pprob$change)
logit.pr.auc <- auc_pr(obs, logit.pprob$change)
logit.roc.auc
logit.pr.auc

## Run elasticnet logit in caret
model.elastic <- train(as.formula(form), 
      metric="ROC", method="glmnet", ## stochastic gradient boosting model
      trControl=tc,
      data=data)
## Model output
model.elastic
## elasticnet predictions
elastic.pprob <- predict(model.elastic, test, type="prob")
elastic.roc.auc <- auc_roc(obs, elastic.pprob$change)
elastic.pr.auc <- auc_pr(obs, elastic.pprob$change)
elastic.roc.auc
elastic.pr.auc

otherprobs <- as.data.frame(elastic.pprob$change, rf.pprob$change,elastic.pprob$change)

colname(otherprobs) <- c('boosting', 'rf', 'elasticnet')

write.csv(otherprobs, file='otherprobs.csv')