validloss <- read.table('data/validationloss.txt')
trainloss <- read.table('data/trainingloss.txt')
epochs <- 1:100
pdf('epoch_performance.pdf', width=9, height=7)
plot(epochs, validloss$V1, type='l', ylim=c(0, .04), ylab='log loss', xlab='Epochs', 
     col='blue', main='Training and Validation Error', cex.main = 2, cex.lab=1.5, cex.axis = 1.1)
lines(epochs, trainloss$V1, col='orange')
legend(x='topleft', c('Training Loss', "Validation Loss"),
       text.col=c('orange', 'blue'),cex=2,
       bty = "n")
dev.off()

