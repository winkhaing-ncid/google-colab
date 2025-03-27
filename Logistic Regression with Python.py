#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression with Python

# The Python code in this Notebook is provided as part of a [Dave on Data](https://www.daveondata.com) crash course on logistic regression with Python.
# 
# The code is built using the mighty [statsmodels](https://www.statsmodels.org/) library. Instructions for installing statsmodels are available [here](https://www.statsmodels.org/stable/install.html).
# 
# This code is provided **as-is** for your use. No warranty for this code should be assumed or is implied.

# ### Load the *Heart* Dataset

# The webinar uses the [Statlog (Heart) Data Set](https://archive.ics.uci.edu/dataset/145/statlog+heart) available from UCI Machine Learning Repository.

# In[1]:


import pandas as pd

# Load the Heart dataset
heart = pd.read_csv('Heart.csv')
heart.head()

# ### Your First Logistic Regression Model

# As both the *HeartDisease* label and *Male* feature are already binary (i.e., the values are either 0 or 1), they can be used directly in creating a logistic regression model. The code below uses a convenient way to specify models based on the R programming language's [formula syntax](https://www.statsmodels.org/dev/example_formulas.html).

# In[2]:


import statsmodels.formula.api as smf

# Craft a logistic regression model to predict HeartDisease based on being Male
heart_model_1 = smf.logit(formula = 'HeartDisease ~ Male', data = heart)

# Train the model from the data
model_1_results = heart_model_1.fit()

# What are the model results?
print(model_1_results.summary())

# ### Your Second Logistic Regression Model

# In[3]:


# A logistic regression model to predict HeartDisease using Male and Age
heart_model_2 = smf.logit(formula = 'HeartDisease ~ Male + Age', data = heart)

# Train the model from the data
model_2_results = heart_model_2.fit()

# What are the model results?
print(model_2_results.summary())

# ### Your Third Logistic Regression Model

# In[4]:


# A logistic regression model to predict HeartDisease using Male, Age, & Angina
heart_model_3 = smf.logit(formula = 'HeartDisease ~ Male + Age + Angina', data = heart)

# Train the model from the data
model_3_results = heart_model_3.fit()

# What are the model results?
print(model_3_results.summary())

# ### Interpreting Coefficients

# In[5]:


from math import exp

# Get the odds ratio for the Male coefficient
print(exp(model_3_results.params['Male']))

# In[6]:


# Get the odds ratio for the Age coefficient
print(exp(model_3_results.params['Age']))

# In[7]:


# Get the odds ratio for the Angina coefficient
print(exp(model_3_results.params['Angina']))
