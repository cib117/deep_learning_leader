## Remove objects from environment
rm(list=ls())

## load dplyr
library(dplyr)

## Load Beger et al. data
load('data/irc_data_mod.rda')

###############################################################################################################
## Prepare data
###############################################################################################################
## Subset data using dplyr
df <- select(irc.data, ccode, country, date, id, failure, i.matl.conf.DIStGOV.l1, 
               i.matl.coop.GOVtGOV.l1, ldr.irregular, ldr.foreign,
               mths.in.power, i.verb.coop.GOVtGOV.l1,
               i.verb.conf.GOVtDIS.l1, i.verb.conf.DIStGOV.l1, 
               IT.NET.USER.P2.l1,
                exclpop.l1, W.knn4.std.ins.l.count.both.l1, SP.DYN.LE00.FE.IN.l1, 
               PARCOMP.l1, NY.GDP.PCAP.KD.l1, eth.rel.l.count.l1, reb.l.count.both.l1, protest.tALL.l1, 
               W.gower.pol.reb.l.count.both.l1, dom.cris.i.count.l1, MS.MIL.XPND.GD.ZS.l1,
                W.centdist.std.opp_resistance.l1, W.centdist.std.repression.l1,
               Amnesty.l1, ProxElection.l1, opp_resistance.l1, SP.POP.TOTL.l1,
               W.centdist.std.opp_resistance.l1,
               W.centdist.std.repression.l1,
                ProxElection.l1, opp_resistance.l1, 
               SP.POP.TOTL.l1,
               intratension.l1, i.protest.tGOV.l1,IT.CEL.SETS.P2.l1,
               NY.GDP.PCAP.KD.l1, AUTOC.l1, ifs__cpi.i, ifsinrsv.i)

v1 <- c('i.matl.conf.DIStGOV.l1', 'i.matl.coop.GOVtGOV.l1', 'mths.in.power',
        'i.verb.coop.GOVtGOV.l1', 'i.verb.conf.GOVtDIS.l1', 'i.verb.conf.DIStGOV.l1',
        'exclpop.l1', 'W.centdist.std.opp_resistance.l1', 'W.centdist.std.repression.l1',
        'ProxElection.l1', 'opp_resistance.l1')

for (i in v1){
  df[,i] <- log10(df[,i]+1)
}

v2 <- c('SP.POP.TOTL.l1', 'NY.GDP.PCAP.KD.l1', 'MS.MIL.XPND.GD.ZS.l1')
for (i in v2){
  df[,i] <- log10(df[,i])
}

v3 <- c('intratension.l1', 'i.protest.tGOV.l1', 'IT.CEL.SETS.P2.l1', 'ifs__cpi.i', 'ifsinrsv.i')
for (i in v3){
  df[,i] <- log(df[,i]+1)
}


## chantge outcome into factor for modelling
#df$failure <- factor(
  #df$failure,
  #levels = c(0,1),
  #labels = c("none", "change"))

## Split data in validation/test sets
calib.start <- as.Date("2010-01-01")
calib.end   <- as.Date("2012-04-30")
test.start  <- calib.end+1
test.end    <- as.Date("2014-03-31")
train <- df[df$date<calib.start, ]
valid <- df[df$date>=calib.start & df$date<=calib.end, ]
test <- df[df$date>=test.start & df$date<=test.end, ]

## Save as RData file for Caret models
save.image('data/prepared_data.RData')

## Write whole df to csv
write.csv(train, 'data/train.csv', row.names=F)
write.csv(valid, 'data/valid.csv', row.names=F)
write.csv(test, 'data/test.csv', row.names=F)

## Normalize the data
## Normalization important for neural networks
df[,6:37] <- apply(df[,6:37], 2, function(x) scale(x))

# Split again with normalized data
train <- df[df$date<calib.start, ]
valid <- df[df$date>=calib.start & df$date<=calib.end, ]
test <- df[df$date>=test.start & df$date<=test.end, ]


## Break into x&y train, valid, test for numpy ingestion
## And write in csv format
write.csv(train[,6:37], 'data/x_train.csv', row.names=F)
write.csv(valid[,6:37], 'data/x_valid.csv', row.names=F)
write.csv(test[6:37], 'data/x_test.csv', row.names=F)
write.csv(train[,5], 'data/y_train.csv', row.names=F)
write.csv(valid[,5], 'data/y_valid.csv', row.names=F)
write.csv(test[,5], 'data/y_test.csv', row.names=F)