#!/usr/bin/env python
# coding: utf-8

# # Cardio Good Fitness Case Study - Descriptive Statistics
# The market research team at AdRight is assigned the task to identify the profile of the typical customer for each treadmill product offered by CardioGood Fitness. The market research team decides to investigate whether there are differences across the product lines with respect to customer characteristics. The team decides to collect data on individuals who purchased a treadmill at a CardioGoodFitness retail store during the prior three months. The data are stored in the CardioGoodFitness.csv file.
# 
# ### The team identifies the following customer variables to study: 
#   - product purchased, TM195, TM498, or TM798; 
#   - gender; 
#   - age, in years; 
#   - education, in years; 
#   - relationship status, single or partnered; 
#   - annual household income ; 
#   - average number of times the customer plans to use the treadmill each week; 
#   - average number of miles the customer expects to walk/run each week; 
#   - and self-rated fitness on an 1-to-5 scale, where 1 is poor shape and 5 is excellent shape.
# 
# ### Perform descriptive analytics to create a customer profile for each CardioGood Fitness treadmill product line.

# In[ ]:


# Load the necessary packages

import numpy as np
import pandas as pd


# In[ ]:


# Load the Cardio Dataset

mydata = pd.read_csv('CardioGoodFitness.csv')


# In[ ]:


mydata.head()


# In[ ]:


mydata.describe(include="all")


# In[ ]:


mydata.info()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

mydata.hist(figsize=(20,30))


# In[ ]:


import seaborn as sns

sns.boxplot(x="Gender", y="Age", data=mydata)


# In[ ]:


pd.crosstab(mydata['Product'],mydata['Gender'] )


# In[ ]:


pd.crosstab(mydata['Product'],mydata['MaritalStatus'] )


# In[ ]:


sns.countplot(x="Product", hue="Gender", data=mydata)


# In[ ]:


pd.pivot_table(mydata, index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'], aggfunc=len)


# In[ ]:


pd.pivot_table(mydata,'Income', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])


# In[ ]:


pd.pivot_table(mydata,'Miles', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])


# In[ ]:


sns.pairplot(mydata)


# In[ ]:


mydata['Age'].std()


# In[ ]:


mydata['Age'].mean()


# In[ ]:


sns.distplot(mydata['Age'])


# In[ ]:


mydata.hist(by='Gender',column = 'Age')


# In[ ]:


mydata.hist(by='Gender',column = 'Income')


# In[ ]:


mydata.hist(by='Gender',column = 'Miles')


# In[ ]:


mydata.hist(by='Product',column = 'Miles', figsize=(20,30))


# In[ ]:


corr = mydata.corr()
corr


# In[ ]:


sns.heatmap(corr, annot=True)


# In[ ]:


# Simple Linear Regression


#Load function from sklearn
from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

y = mydata['Miles']
x = mydata[['Usage','Fitness']]

# Train the model using the training sets
regr.fit(x,y)


# In[ ]:


regr.coef_


# In[ ]:


regr.intercept_


# In[ ]:


# MilesPredicted = -56.74 + 20.21*Usage + 27.20*Fitness

