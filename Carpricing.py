#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


car = pd.read_csv('cardata.csv')


# In[4]:


car


# In[5]:


car.head()


# In[6]:


car.tail()


# In[7]:


car.describe()


# # Checking for missing or null values

# In[8]:


car.isna()


# In[9]:


car.isnull().sum()


# In[10]:


car.shape


# In[11]:


print(car['Seller_Type'].unique())
print(car['Transmission'].unique())
print(car['Owner'].unique())
print(car['Fuel_Type'].unique())


# In[12]:


car.columns


# In[13]:


final_dataset=car[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[14]:


final_dataset.head()


# In[15]:


final_dataset['Current_Year']=2021


# In[16]:


final_dataset.head()


# In[17]:


final_dataset['No_Of_Year']=final_dataset['Current_Year']-final_dataset['Year']


# In[18]:


final_dataset.head()


# In[19]:


final_dataset.drop('Current_Year', axis=1, inplace=True)


# In[20]:


final_dataset.head()


# In[21]:


final_dataset.drop('Year',axis=1, inplace=True)


# In[22]:


final_dataset.head()


# In[23]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[24]:


final_dataset.head()


# In[25]:


final_dataset.corr()


# In[26]:


import seaborn as sns


# In[27]:


sns.pairplot(final_dataset)


# In[28]:


corrmath = final_dataset.corr()
top_corr_features=corrmath.index
plt.figure(figsize=(25,25))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[29]:


final_dataset.head()


# In[30]:


#Independent and dependent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[31]:


X.head()


# In[32]:


y.head()


# In[33]:


##Finding Features importances
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[34]:


print(model.feature_importances_)


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[36]:


X_train


# In[37]:


X_train.shape


# In[38]:


y_train.shape


# In[39]:


from sklearn.ensemble import RandomForestRegressor
rf_random =RandomForestRegressor()


# In[40]:


n_estimators = [int(x) for x in np.linspace(start = 100,stop=1200,num = 12)]
print(n_estimators)


# In[41]:


#Randomized SearchCV
# Number of tress in random Forest
n_estimators = [int(x) for x in np.linspace(start = 100,stop=1200,num = 12)]
# Number of features to considers at every split
max_features = ['auto','sqrt']
#Maximum number of Levels in tree
max_depth =[int(x)for x in np.linspace(5,30, num=6)]
#max_depth.append(None)
#Minimum number of samples required to split a node
min_samples_split =[2,5,10,15,100]
#Minimum number of samples required at each Leaf Node
min_samples_leaf =[1,2,5,10]


# In[42]:


max_features = ['auto','sqrt']
print(max_features)


# In[43]:


max_depth = [int(x)for x in np.linspace(5,30,num=6)]
print(max_depth)


# In[44]:


min_samples_split = [2,5,10,15,100]
print(min_samples_split)


# In[45]:


min_samples_leaf = [1,2,5,10]
print(min_samples_leaf)


# In[46]:


from sklearn.model_selection import RandomizedSearchCV


# In[47]:


#Create The random Grid
random_grid ={'n_estimators' :n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf}
               
              
print(random_grid)


# In[48]:


#Use the random grid to search for best hyperparameters
#Create the model to tune
rf = RandomForestRegressor()


# In[49]:


rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,scoring='neg_mean_squared_error',n_iter =10,cv = 5,verbose=2,random_state=42,n_jobs=1)


# In[50]:


rf_random.fit(X_train,y_train)


# In[51]:


predictions = rf_random.predict(X_test)


# In[52]:


predictions


# In[53]:


sns.distplot(y_test-predictions)


# In[54]:


plt.scatter(y_test,predictions)


# In[56]:


import pickle
# open a file, where you want to store the data
file = open('random_forest_regression_model.pk1','wb')

#dump information to that file
pickle.dump(rf_random,file)


# In[ ]:




