""" Created on Wed Jan  9 20:45:25 2019 @author: Izadi """

""" Please run

 logitreg_grid.fit(Xtrain, ytrain)

befor the last 10 print commands separately. Since it is a little heavy , 30 seconds. 

Also all libraries from sklearn are necessary.  """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math 

from sklearn.preprocessing import StandardScaler, Normalizer, scale
from sklearn.metrics import confusion_matrix, log_loss, auc, roc_curve, roc_auc_score, recall_score, precision_recall_curve
from sklearn.metrics import make_scorer, precision_score, fbeta_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedShuffleSplit, GridSearchCV

from sklearn.linear_model import LogisticRegression

%matplotlib inline


from pylab import rcParams
rcParams['figure.figsize'] = 17, 12

def sigmoid(x):      # graph of Logistic regression function. This is just for fun, Not necessary.
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a
x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
plt.plot(x,sig)
plt.show()

crd=pd.read_csv(r'C:\Users\Izadi\Desktop\Farzali_sampledata4.csv')

type(crd)
crd.shape
crd.head()
crd.tail()

A= crd.T        #Transforming (rows, columns) into (columns,rows) to see hiddend head title
A

crd.columns = [x.lower() for x in crd.columns]  #x.lower lower v variables
crd.rename(columns = {'class': 'fraud'}, inplace = True)
crd.fraud.value_counts(dropna = False)

crd.groupby('fraud').amount.mean()

print("Number of class:"+ str(len(crd.index)))
crd['fraud'].value_counts() 
print( "fraud:" + "50 out of 5000")

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

sns.countplot(crd['fraud'])

#Data wrangling 
crd.info()
crd.describe() 
crd.isnull() 

cr = crd['fraud']
type(cr)
len(cr)
n=cr.count()  # of non NaN s which there isn't any
b=cr.unique()
len(b)

from pylab import rcParams
rcParams['figure.figsize'] = 20, 12

crd["time"].plot.hist(figsize=(60, 10))       #time
crd["amount"].plot.hist(figsize=(60, 10))     #amount
crd["v9"].plot.hist(figsize=(60,10))
crd["v11"].plot.hist(figsize=(60,10))
crd["v26"].plot.hist(figsize=(60,10))        # 9 10 11  26 look balanced

for index in crd.columns[1:8]:
    crd[index].plot.hist(figsize=(13,10))  #v1 to v6

for index in crd.columns[8:15]:             #v7 to v14
    crd[index].plot.hist(figsize=(13,10))
    
for index in crd.columns[15:21]:                  #v15 to v21
    crd[index].plot.hist(figsize=(13,10))
    
for index in crd.columns[21:29]                   #v21 to v28
    crd[index].plot.hist(figsize=(15,7)) 
    
for index in crd.columns[1:29]:                  # all from v1 to v28
    crd[index].plot.hist(figsize=(30,10))   
    
def correlation_matrix(crd):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(crd.corr('kendall'), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('crd Feature Correlation')
    labels=crd.columns
    ax1.set_xticklabels(labels,fontsize=20)
    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
correlation_matrix(crd)
  
crd.corr(method='pearson')        # By default corr() is pearson

crd.corr(method='spearman')

crd.corr(method='kendall')

z1=crd.iloc[:,:]               # [:,:] All rows and columns
z1.head()

x1 = crd.iloc[:, 1:7]                #creditcard.iloc[:,1:7]=creditcard.iloc[:,1:-24]
x1 = crd.iloc[:, 1:-24]              #This all colums from v1 to 31-24-1 = 6 to v6  (-1 for time)
x2 = crd.iloc[:, 7:-17]             #From v7 to w13 i.e., 31-17=14 to v13  same as iloc[: , 7:14]

def correlation_matrix(x1):               # Corr amoung v1 to v6
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 20)
    cax = ax1.imshow(x1.corr('kendall'), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('x1 Feature Correlation')
    labels=x1.columns
    ax1.set_xticklabels(labels,fontsize=0)
    ax1.set_yticklabels(labels,fontsize=20)
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
correlation_matrix(x1)

x1.corr('pearson')          # numerical corr of three method for v1 to v6                                
x1.corr('spearman')
x1.corr('kendall')

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10


sns.pairplot(crd, x_vars=['v1','v2','v3', 'v4'], y_vars='fraud' , size=7, aspect=0.7)  # paiplot 

sns.pairplot(crd, x_vars=['v1','v2','v3','v4'], y_vars='fraud' , size=7, aspect=0.7 , kind='reg') #pair and regression


#Normalize the 'amount' N(0,1)
scaler = StandardScaler()

crd['amount'] = scaler.fit_transform(crd['amount'].values.reshape(-1, 1)) #Normalize

X = crd.iloc[:, :-1]        # All columns but fraud
y = crd.iloc[:, -1]         # Only frauf column

# 1. Split data   for Training and Testing   Train=75%  and Test = 25%
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = .25, stratify = y, random_state = 875)

logitreg_parameters = {'C': np.power(10.0, np.arange(-3, 3))}
logitreg = LogisticRegression(verbose = 3, warm_start = True)
logitreg_grid = GridSearchCV(logitreg, param_grid = logitreg_parameters, scoring = 'roc_auc', n_jobs = 70)

logitreg_grid.fit(Xtrain, ytrain)   # logitreg_grid.best_params_ ; logitreg_grid.best_estimator_

# 1. on OVER-Sampled TRAINing data
print("\n The recall score on Training data is: 0.6756756756756757 ") #The value may change due to randomizatios 
print(recall_score(ytrain, logitreg_grid.predict(Xtrain)))    
print("\n The precision score on Training data is: 0.8064516129032258 ")
print(precision_score(ytrain, logitreg_grid.predict(Xtrain)))   
# on the separated TEST data
print("\n The recall score on Test data is: 0.38461538461538464 ")
print(recall_score(ytest, logitreg_grid.predict(Xtest)))   
print("\n The precision score on Test data is: 1.0 ")
print(precision_score(ytest, logitreg_grid.predict(Xtest)))   
print("\n The Confusion Matrix on Test data is: [[1250    0] [   8    5]]")
print(confusion_matrix(ytest, logitreg_grid.predict(Xtest)))

# 2. If we don't do Over-sampling, what will happen for training?
ytrain_pred_probas = logitreg_grid.predict_proba(Xtrain)[:, 1]   # prob of predict as 1
fpr, tpr, thresholds = roc_curve(ytrain, ytrain_pred_probas)   # precision_recall_curve
roc = pd.DataFrame({'FPR':fpr,'TPR':tpr,'Thresholds':thresholds})
_ = plt.figure()
plt.plot(roc.FPR, roc.TPR)
plt.axvline(0.1, color = '#00C851', linestyle = '--')
plt.xlabel("FPR")
plt.ylabel("TPR")





