#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/Student_Marks.csv')


# In[3]:


data


# In[4]:


data["number_courses"].value_counts()


# In[5]:


figure = px.scatter(data_frame=data, x = "number_courses", 
                    y = "Marks", size = "time_study", 
                    title="Number of Courses and Marks Scored")
figure.show()


# In[6]:


figure = px.scatter(data_frame=data, x = "time_study", 
                    y = "Marks", size = "number_courses", 
                    title="Time Spent and Marks Scored", trendline="ols")
figure.show()


# In[7]:


correlation = data.corr()
print(correlation["Marks"].sort_values(ascending=False))


# In[8]:


x = np.array(data[["time_study", "number_courses"]])
y = np.array(data["Marks"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[9]:


model = LinearRegression()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[11]:


features = np.array([[4.8, 6]])
model.predict(features)


# In[ ]:




