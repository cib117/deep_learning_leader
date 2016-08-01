#######################################################################
##  Filename: activation_functions.r
##  Purpose: Plots to demonstate different activation functions
##  Uses data from: Randomly generated data
##  Output to: plots/activation_func.pdf
##  Last Edited: 20 July 2016
##  Christopher Boylan, Penn State University
#######################################################################
## Prepare data
#######################################################################
## Generate x values
x <- seq(-2,2,.01)
## Different functions used in NN
logit <- 1/(1+exp(-x)) ## sigmoid/logistic
tanh <- tanh(x) ## hyperbolic tangent
relu <- sapply(x,function(z) max(0,z)) ## rectified linear unit ReLU
#######################################################################
## Plot different activation functions
#######################################################################
pdf('plots/activation_func.pdf', width=9, height=7)
plot(x,relu, ylab='f(x)',xlab='x',main="Activation Functions", ylim=c(-2,2), type='l', lwd=1.5, col='blue')
lines(x, logit, type='l', lwd=1.5, col='green')
lines(x, tanh, type='l', lwd=1.5, col='purple')
legend(x='topleft', c('ReLU', 'Logistic', 'Tanh'),
       text.col=c('blue', 'green', 'purple'),
       bty = "n") ## Add legend
dev.off()
