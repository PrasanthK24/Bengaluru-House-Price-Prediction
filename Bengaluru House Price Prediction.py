#!/usr/bin/env python
# coding: utf-8

# # Predicting House Price in Bengaluru

# Dataset is downloaded from here: https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data

# ## Import Libraries

# In[4]:


import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,VotingClassifier
from sklearn.linear_model import SGDRegressor, LinearRegression, BayesianRidge, Lasso, HuberRegressor, ElasticNetCV


# ## Load Dataset 

# In[6]:


df1 = pd.read_csv(r"D:\Data Science\Projects\Bengaluru House Price Prediction\archive\Bengaluru_House_Data.csv")
df1.head()


# In[7]:


df1.shape


# In[8]:


df1.columns


# In[9]:


df1['area_type'].value_counts()


# ## Droping features that are not required to build the model 

# In[10]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# ## Data Cleaning 

# In[11]:


df2.isnull().sum()


# In[12]:


df3 = df2.dropna()
df3.isnull().sum()


# In[13]:


df3.shape


# ## Feature Engineering 

# In[14]:


df3['size'].unique()


# ### Adding new feature for bhk in the dataset 

# In[15]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# ### Explore total_sqft feature 

# In[16]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[18]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[19]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(5)


# In[20]:


df4.loc[30]


# In[21]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# Examine locations which is a categorical variable. Need to apply dimensionality reduction technique to reduce the number of locations.

# In[22]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# ## Dimensionality Reduction 

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later, when we do one hot encoding, it will help us with having fewer dummy columns.

# In[23]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[24]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[25]:


df5.head(10)


# In[26]:


df5.shape


# Check the above data points. We have 6 bhk apartment with 1020 sqft. This is a clear data error that can be removed safely.

# In[27]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# ### Outlier Removal Using Standard Deviation and Mean 

# In[28]:


df6.price_per_sqft.describe()


# Here we find that min price per sqft is 267 Rs/sqft whereas max is 176470 Rs/sqft. This shows a wide variation in property prices. We should remove outliers per location using mean and standard deviation.

# In[29]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# Let's check for a given location how does the 2 BHK and 3 BHK property prices look like

# In[30]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
   
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[31]:


plot_scatter_chart(df7,"Hebbal")


# #### Remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment 

# In[32]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[33]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[34]:


plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# ### Outlier Removal Using Bathrooms Feature 

# In[35]:


df8.bath.unique()


# In[36]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[37]:


df8[df8.bath>10]


# It is unusual to have 2 more bathrooms than number of bedrooms in a home.

# In[38]:


df8[df8.bath>df8.bhk+2]


# In[39]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[40]:


df9.head()


# In[41]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# ## Use One Hot Encoding For Location

# In[42]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[43]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[44]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# ## Model 

# In[45]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[46]:


X.shape


# In[47]:


Y = df12.price
Y.head(3)


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[49]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# ### Use K-Fold Cross-Validation to measure accuracy of our Linear Regression model

# In[51]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, Y, cv=cv)


# In[53]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,Y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,Y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,Y)


# Based on the above results we can say that Linear Regression gives the best score.

# In[54]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[55]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[56]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[57]:


predict_price('Indira Nagar',1000, 3, 3)


# ### Export the tested model to a pickle file 

# In[59]:


import pickle
with open(r'D:\Data Science\Projects\Bengaluru House Price Prediction\Bengaluru_House_Price_Prediction_Model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# ### Export location and column information to a file that will be useful later for our prediction application

# In[61]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open(r"D:\Data Science\Projects\Bengaluru House Price Prediction\columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




