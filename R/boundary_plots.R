#######################################################################
##  Filename: activation_functions.r
##  Purpose: Plots to demonstate how NN can
##  learn complex functions to predict
##  outcomes that are not linearly separable
##  Uses data from: Randomly generated data
##  Requires packages: ggplot2
##  Output to: plots/NNboundary_func.pdf
##  Last Edited: 20 July 2016
##  Christopher Boylan, Penn State University
#######################################################################
## Load required packages
#######################################################################
rm(list=ls())
set.seed(1804)
library(ggplot2)
#######################################################################
## Multiple plot function
## Obtained from:
## http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
#######################################################################
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
#######################################################################
## Single layer RELU model
## This borrows heavily from:
## Build your own neural network classifier in R
## https://www.r-bloggers.com/build-your-own-neural-network-classifier-in-r/
#######################################################################
## Prepare data
N <- 200 # number of points per class
D <- 2 # dimensionality
K <- 2 # number of classes
X <- data.frame() # data matrix (each row = single example)
y <- data.frame() # class labels
## Generate two class spiral dataset
for (j in (1:K)){
  r <- seq(0.05,1,length.out = N) # radius
  t <- seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
  Xtemp <- data.frame(x =r*sin(t) , y = r*cos(t)) 
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  y <- rbind(y, ytemp)
}

data <- cbind(X,y)
colnames(data) <- c(colnames(X), 'label')
x_min <- min(X[,1])-0.2; x_max <- max(X[,1])+0.2
y_min <- min(X[,2])-0.2; y_max <- max(X[,2])+0.2
X <- as.matrix(X)
Y <- matrix(0, N*K, K)

for (i in 1:(N*K)){
  Y[i, y[i,]] <- 1
}

#######################################################################
## nnet function
#######################################################################
nnet <- function(X, Y, step_size = 0.5, reg = 0.001, h = 10, niteration){
  # X = Feature matrix
  # Y = Vector of outcomes
  # step_size = step_size for gradient descent algorithm
  # reg = regularization parameter
  # h = number of hidden nodes
  # niteration = gradient descent iterations
  # get dim of input
  N <- nrow(X) # number of examples
  K <- ncol(Y) # number of classes
  D <- ncol(X) # dimensionality
  # get dim of input
  N <- nrow(X) # number of examples
  K <- ncol(Y) # number of classes
  D <- ncol(X) # dimensionality
  
  # initialize parameters randomly
  W <- 0.01 * matrix(rnorm(D*h), nrow = D) ## Weights for xW part
  b <- matrix(0, nrow = 1, ncol = h) ## intercept/bias term
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h) ## Weights for aW part
  b2 <- matrix(0, nrow = 1, ncol = K)
  
  # gradient descent loop to update weight and bias
  for (i in 0:niteration){
      # hidden layer, ReLU activation
      hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))
      hidden_layer <- matrix(hidden_layer, nrow = N)
      # class score
      scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)
      
      # compute and normalize class probabilities
      exp_scores <- exp(scores)
      probs <- exp_scores / rowSums(exp_scores)
      
      # compute the loss: sofmax and regularization
      corect_logprobs <- -log(probs)
      data_loss <- sum(corect_logprobs*Y)/N
      reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) ## L2 regularization
      loss <- data_loss + reg_loss
      # check progress
      if (i%%1000 == 0 | i == niteration){
          print(paste("iteration", i,': loss', loss))}
      
      # compute the gradient on scores
      dscores <- probs-Y
      dscores <- dscores/N
      
      # backpropate the gradient to the parameters
      dW2 <- t(hidden_layer)%*%dscores
      db2 <- colSums(dscores)
      # next backprop into hidden layer
      dhidden <- dscores%*%t(W2)
      # backprop the ReLU non-linearity
      dhidden[hidden_layer <= 0] <- 0
      # finally into W,b
      dW <- t(X)%*%dhidden
      db <- colSums(dhidden)
      
      # add regularization gradient contribution
      dW2 <- dW2 + reg *W2
      dW <- dW + reg *W
      
      # update parameter
      W <- W-step_size*dW
      b <- b-step_size*db
      W2 <- W2-step_size*dW2
      b2 <- b2-step_size*db2
  }
  return(list(W, b, W2, b2))
}

nnetPred <- function(X, para = list()){
    W <- para[[1]]
    b <- para[[2]]
    W2 <- para[[3]]
    b2 <- para[[4]]
    
    N <- nrow(X)
    hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  ## Recitifer wrapped in matrix multiplication
    hidden_layer <- matrix(hidden_layer, nrow = N) ## Convert into matrix
    scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T) ## Get scores
    predicted_class <- apply(scores, 1, which.max) ## Class with highest score is final prediction
    
    return(predicted_class)  
}



nnetPred <- function(X, para = list()){
  W <- para[[1]]
  b <- para[[2]]
  W2 <- para[[3]]
  b2 <- para[[4]]
  
  N <- nrow(X)
  hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T)) 
  hidden_layer <- matrix(hidden_layer, nrow = N)
  scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T) 
  predicted_class <- apply(scores, 1, which.max)
  
  return(predicted_class)  
}
#######################################################################
## Plot different Neural Network models of increasing complexity
#######################################################################
## NN with 1 node in hidden layer
nnet.model1 <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=1, niteration = 6000)
predicted_class1 <- nnetPred(X, nnet.model1)
print(paste('training accuracy:',mean(predicted_class == (y))))
hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by =hs)))
Z1 <- nnetPred(grid, nnet.model1)
nn1 <- (ggplot()+
geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z1)), alpha = 0.2, show.legend = F)+
geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + theme_bw(base_size = 15) +
ggtitle('NN with 1 node in hidden layer') +
coord_fixed(ratio = 0.8) +
scale_color_manual(values=c("blue", "orange"))+
scale_fill_manual(values=c("blue", "orange"))+
theme(axis.ticks=element_blank(),
panel.grid.major = element_blank(),
panel.grid.minor = element_blank(),
axis.text=element_blank(),
axis.title=element_blank(),
legend.position = 'none',
panel.border = element_blank(),
plot.title = element_text(face="bold", size=24)))

nn1

## NN with 3 nodes in hidden layer
nnet.model2 <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=3, niteration = 6000)
predicted_class2 <- nnetPred(X, nnet.model2)
print(paste('training accuracy:',mean(predicted_class2 == (y))))
hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by =hs)))
Z2 <- nnetPred(grid, nnet.model2)
nn3 <- ggplot()+
geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z2)), alpha = 0.2, show.legend = F)+
geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + theme_bw(base_size = 15) +
ggtitle('NN with 3 nodes in hidden layer') +
coord_fixed(ratio = 0.8) +
scale_color_manual(values=c("blue", "orange"))+
scale_fill_manual(values=c("blue", "orange"))+
theme(axis.ticks=element_blank(),
panel.grid.major = element_blank(),
panel.grid.minor = element_blank(),
axis.text=element_blank(),
axis.title=element_blank(),
legend.position = 'none',
panel.border = element_blank(),
plot.title = element_text(face="bold", size=24))

nn3

## NN with 10 nodes in hidden layer
nnet.model3 <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=10, niteration = 6000)
predicted_class3 <- nnetPred(X, nnet.model3)
print(paste('training accuracy:',mean(predicted_class3 == (y))))
hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by =hs)))
Z3 <- nnetPred(grid, nnet.model3)

nn10 <- ggplot()+
geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z3)), alpha = 0.2, show.legend = F)+
geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + theme_bw(base_size = 15) +
ggtitle('NN with 10 nodes in hidden layer') +
coord_fixed(ratio = 0.8) +
scale_color_manual(values=c("blue", "orange"))+
scale_fill_manual(values=c("blue", "orange"))+
theme(axis.ticks=element_blank(),
panel.grid.major = element_blank(),
panel.grid.minor = element_blank(),
axis.text=element_blank(),
axis.title=element_blank(),
legend.position = 'none',
panel.border = element_blank(),
plot.title = element_text(face="bold", size=24))

nn10
## NN with 20 nodes in hidden layer
nnet.model4 <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=20, niteration = 6000)
predicted_class4 <- nnetPred(X, nnet.model4)
print(paste('training accuracy:',mean(predicted_class4 == (y))))
hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by =hs)))
Z4<- nnetPred(grid, nnet.model4)
nn20 <- ggplot()+
geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z4)), alpha = 0.2, show.legend = F)+
geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + theme_bw(base_size = 15) +
ggtitle('NN with 20 nodes in hidden layer') +
coord_fixed(ratio = 0.8) +
scale_color_manual(values=c("blue", "orange"))+
scale_fill_manual(values=c("blue", "orange"))+
theme(axis.ticks=element_blank(),
panel.grid.major = element_blank(),
panel.grid.minor = element_blank(),
axis.text=element_blank(),
axis.title=element_blank(),
legend.position = 'none',
panel.border = element_blank(),
plot.title = element_text(face="bold", size=24))

nn20
## Save plot as pdf
pdf('plots/NNboundary.pdf' ,height=12, width =12 )
multiplot(nn1, nn10, nn3, nn20, cols=2)
dev.off()