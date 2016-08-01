# Import required libraries, functions
from keras.models import Sequential ## Sequential is required to build model in keras
from keras.layers.core import Dense, Dropout, Activation ## different pieces for neural net models
from keras.optimizers import SGD ## Stochastic gradient descent to learn optimal parameters
from keras.callbacks import History ## History used to get training and validation performance across epochs
history = History() ## history to get training and validation performance across epochs
import numpy as np ## numpy for data ingestion
from keras.utils import np_utils ## used to convert outcome into desired format for Keras
from keras.constraints import maxnorm ## Maxnorm contraint to prevent Weights getting very large
from sklearn.metrics import (precision_score, recall_score, ## precision, recall
                             f1_score, accuracy_score, roc_auc_score, average_precision_score) ## F1,ROC-AUC,PR-AUC

# set seed
np.random.seed(1804) 

# Import IL data (prepared in R since the replication data in RData format)
X_train = np.genfromtxt('data/x_train.csv', delimiter=',', skip_header=1)
X_valid = np.genfromtxt('data/x_valid.csv',  delimiter=',', skip_header=1)
X_test = np.genfromtxt('data/x_test.csv',  delimiter=',', skip_header=1)
y_train = np.genfromtxt('data/y_train.csv', skip_header=1)
y_valid = np.genfromtxt('data/y_valid.csv', skip_header=1)
y_test= np.genfromtxt('data/y_test.csv', skip_header=1)

# The fit functions expects this labels to be encoded as one-hot vectors.
# In this case, this means we want label matrices with n_train, n_test, n_valid rows, each row being
# [1, 0] (class 0) or [0, 1] (class 1).
# Use util function to convert our labels vector to this format
Y_train = np_utils.to_categorical(y_train)
Y_valid = np_utils.to_categorical(y_valid)
Y_test = np_utils.to_categorical(y_test)

###################################
# Model (single hidden layer, 40 hidden nodes, use dropout 0.5 for regularization)
###################################
# Initialize model object
model = Sequential()
# Dense(64) is a fully-connected layer with 40 hidden units.
# in the first layer, you must specify the expected input data shape
model.add(Dense(40, input_dim=X_train.shape[1], init='uniform')) # X_train.shape[1] == 32 here
model.add(Activation('relu')) # ReLU activation function
model.add(Dropout(0.5)) # Use Dropout to regularize model
model.add(Dense(Y_train.shape[1], init='uniform')) # y_train.shape[1] == 2 here
model.add(Activation('sigmoid'))
## Stochastic gradient descent parameters, just copied someone elses
## These can be searched over and optimized also
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
## Choose loss function to measure error between predictions and ground truth
model.compile(loss='binary_crossentropy', optimizer=sgd)
## Start model estimation
model.fit(X_train, Y_train, nb_epoch=100, batch_size=32, validation_data=(X_valid, Y_valid), show_accuracy=True, callbacks=[history])
## Get traina dnd validation loss across epochs
trainloss = history.history['loss']
validloss = history.history['val_loss']
## Write training loss to plot in R
tloss = open('trainingloss.txt', 'w')
for i in trainloss:
    tloss.write(str(i)+'\n')

tloss.close()
## Write validation loss to plot in R
vloss = open('validationloss.txt', 'w')
for i in validloss:
    vloss.write(str(i)+'\n')

vloss.close()
## Get validation predictions
y_valid_pred = model.predict(X_valid)
change_pred = [i[1] for i in y_valid_pred]
## Calculate ROC-AUC and PR-AUC
roc_auc_score(y_valid, change_pred)
average_precision_score(y_valid, change_pred)
# Evaluate test performance
y_test_pred = model.predict(X_test)
tchange_pred = [i[1] for i in y_test_pred]
average_precision_score(y_test, tchange_pred)
roc_auc_score(y_test, tchange_pred)
## We can that running too many epochs causes overfitting
######################################################################
# set seed
np.random.seed(1804) 
###################################
# Model (single hidden layer, 40 hidden nodes, use dropout 0.5 for regularization)
# Reduced number of epochs here
###################################
# Initialize model object
model = Sequential()
# Dense(64) is a fully-connected layer with 40 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 32-dimensional vectors.
model.add(Dense(40, input_dim=X_train.shape[1], init='uniform')) # X_train.shape[1] == 32 here
model.add(Activation('sigmoid')) # ReLU activation function
#model.add(Dropout(0.5)) # Use Dropout to regularize model
model.add(Dense(Y_train.shape[1], init='uniform')) # y_train.shape[1] == 2 here
model.add(Activation('sigmoid')) ## Use sigmoid to get classess/predicted probabilities

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, nb_epoch=10, batch_size=32, validation_data=(X_valid, Y_valid), show_accuracy=True)

y_valid_pred = model.predict(X_valid)
change_pred = [i[1] for i in y_valid_pred]
roc_auc_score(y_valid, change_pred)
average_precision_score(y_valid, change_pred)
# Evaluate test performance
y_test_pred = model.predict(X_test)
tchange_pred = [i[1] for i in y_test_pred]
average_precision_score(y_test, tchange_pred)
roc_auc_score(y_test, tchange_pred)
# Write predictions
preds = open('data/predictionsRELU40.txt', 'w')
for i in tchange_pred:
    preds.write(str(i)+'\n')

preds.close()

# set seed
np.random.seed(1804) 
###################################
# Model (3 hidden layers, 40, 20, 20 hidden nodes,
# use dropout 0.5 for regularization)
# Reduced number of epochs here
###################################
# Initialize model object
model = Sequential()
# Dense(40) is a fully-connected layer with 40 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 32-dimensional vectors.
model.add(Dense(40, input_dim=X_train.shape[1], init='uniform', W_constraint = maxnorm(5))) # X_train.shape[1] == 32 here
model.add(Activation('relu')) # ReLU activation function
model.add(Dropout(0.5))
model.add(Dense(20, init='uniform', W_constraint = maxnorm(5))) # X_train.shape[1] == 32 here
model.add(Activation('relu')) # ReLU activation function
model.add(Dropout(0.5))
model.add(Dense(20, init='uniform', W_constraint = maxnorm(5))) # X_train.shape[1] == 32 here
model.add(Activation('relu')) # ReLU activation function
model.add(Dropout(0.5))
model.add(Dense(Y_train.shape[1], init='uniform')) # y_train.shape[1] == 2 here
model.add(Activation('sigmoid')) ## Use sigmoid to get classess/predicted probabilities

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, nb_epoch=25, batch_size=32, validation_data=(X_valid, Y_valid), show_accuracy=True)

y_valid_pred = model.predict(X_valid)
change_pred = [i[1] for i in y_valid_pred]
roc_auc_score(y_valid, change_pred)
average_precision_score(y_valid, change_pred)
# Evaluate test performance
y_test_pred = model.predict(X_test)
tchange_pred = [i[1] for i in y_test_pred]
average_precision_score(y_test, tchange_pred)
roc_auc_score(y_test, tchange_pred)
# performance doesn't increase with multiple layers
# Need to explore more mutlilayer models using a random or grid search

