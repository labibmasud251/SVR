#data preprocessing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_independantvar=LabelEncoder()

dataset=pd.read_excel('Friesian.xlsx')
#extracting Feature columns that is independant variable matrix
#we could use -1 instead of 3 same result will come
dataset.head()

for column in dataset.columns:
    if dataset[column].dtype == type(object):
        dataset[column] = labelencoder_independantvar.fit_transform(dataset[column])
# =============================================================================
independantFeatures=dataset.iloc[:, :-1].values
depandantvar=dataset.iloc[:,6].values
# #special case for showing  missing nan vlue
# np.set_printoptions(threshold=np.nan)
# #handling Missing value by importing Imputer from SKLern
# from sklearn.preprocessing import Imputer
# imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
# myimputer=imputer.fit(independantFeatures[:, 1:3])
# =============================================================================
#now transform the missingvalue with columns mean value and replace
#with nan value

# =============================================================================
# independantFeatures[:, 1:3]=myimputer.transform(independantFeatures[:, 1:3])
# 
# #hadling  categorical varilbe encoding
#this will fit and trnsform encoded value to independant features country column
independantFeatures[:,0]=labelencoder_independantvar.fit_transform(independantFeatures[:,0])
#print(list(independentFeatures))
 
#onehotencoder_independantvar=OneHotEncoder(categorical_features=[0])
#independantFeatures=onehotencoder_independantvar.fit_transform(independantFeatures).toarray()
 
# =============================================================================
# dealing with dependant variable categorical data


# =============================================================================
labelencoder_dependantvar=LabelEncoder()
depandantvar=labelencoder_dependantvar.fit_transform(depandantvar)
#print(list(depandantvar))
# 
# =============================================================================
#splitting  the dataset into Training  set and Test set
from sklearn.cross_validation import train_test_split
independantVartrain,independantVartest,dependantVartrain,dependantVartest=train_test_split(independantFeatures,depandantvar, test_size=0.2,random_state=0)

#Feature scaling

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import svm

scIvar=StandardScaler()
independantVartrain=scIvar.fit_transform(independantVartrain)
independantVartest=scIvar.fit_transform(independantVartest)
print(dependantVartrain)
dependantVartrain = np.reshape(dependantVartrain, (-1, 2))
dependantVartest = np.reshape(dependantVartest, (-1, 2))
print(dependantVartrain)
dependantVartrain=scIvar.fit_transform(dependantVartrain)
dependantVartest=scIvar.transform(dependantVartest)
dependantVartrain = np.reshape(dependantVartrain, (-1, 1))
dependantVartest = np.reshape(dependantVartest, (-1, 1))
print(dependantVartrain)
print(independantVartrain)

import statsmodels.api as sm
import matplotlib.pyplot as plt

est = sm.OLS(dependantVartrain, independantVartrain).fit()

print(est.summary())

#print(len(independantVartrain), len(dependantVartrain))

clf = LinearRegression()
#independantVartrain = np.argmax(independantVartrain, axis=1)
#independantVartest = np.argmax(independantVartest, axis=1)
clf.fit(dependantVartrain, independantVartrain)
confidence = clf.score(dependantVartest, independantVartest)
print(confidence)

print(dependantVartest, independantVartest)
print(dependantVartrain, independantVartrain)

clf = svm.SVR(kernel='linear')
independantVartrain = np.argmax(independantVartrain, axis=1)
independantVartest = np.argmax(independantVartest, axis=1)
clf.fit(dependantVartrain, independantVartrain)
confidence1 = clf.score(dependantVartest, independantVartest)
print(confidence1)

clf = svm.SVR(kernel='poly')
#independantVartrain = np.argmax(independantVartrain, axis=1)
#independantVartest = np.argmax(independantVartest, axis=1)
clf.fit(dependantVartrain, independantVartrain)
confidence2 = clf.score(dependantVartest, independantVartest)
print(confidence2)


clf = svm.SVR(kernel='rbf')
#independantVartrain = np.argmax(independantVartrain, axis=1)
#independantVartest = np.argmax(independantVartest, axis=1)
clf.fit(dependantVartrain, independantVartrain)
confidence3 = clf.score(dependantVartest, independantVartest)
print(confidence3)


clf = svm.SVR(kernel='sigmoid')
#independantVartrain = np.argmax(independantVartrain, axis=1)
#independantVartest = np.argmax(independantVartest, axis=1)
clf.fit(dependantVartrain, independantVartrain)
confidence4 = clf.score(dependantVartest, independantVartest)
print(confidence4)

#for k in ['linear','poly','rbf','sigmoid']:
#    clf = svm.SVR(kernel=k)
#    independantVartrain = np.argmax(independantVartrain, axis=1)
#    independantVartest = np.argmax(independantVartest, axis=1)
#    clf.fit(dependantVartrain, independantVartrain)
#    confidence1 = clf.score(dependantVartest, independantVartest)
#    print(k,confidence1)
