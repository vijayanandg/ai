#!/usr/bin/env python
# coding: utf-8

# In[14]:


import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('pandas: {}'.format(pandas.__version__))


# In[18]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[24]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url,names=names)
print(dataset.shape)



# In[26]:


print(dataset.describe())    


# In[29]:


print(dataset.groupby("class").min())


# In[33]:


dataset.plot(kind='box',subplots=True,sharex=False,sharey=False,layout=(4,4))
plt.show()


# In[34]:


dataset.hist()
plt.show()


# In[35]:


scatter_matrix(dataset)
plt.show()


# In[46]:


array = dataset.values
X = array[:,0:4]
Y = array[:,4]


# In[49]:


validation_size = 0.2
seed = 6
X_train,Y_train,X_test,Y_test = model_selection.train_test_split(X,Y,test_size = validation_size, random_state=seed)


# In[58]:


scoring = 'accuracy'
seed = 6
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))


# In[ ]:


results = []
names = []
seed = 6
for name,model in models:
    kfold = model_selection.KFold(n_splits = 10, shuffle=True, random_state = seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)


# In[ ]:




