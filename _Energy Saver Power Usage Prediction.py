
# coding: utf-8

# # Energy Saver(Street Light Controller) Power Usage Prediction
# 

# ## Description
# 
# A company has implemented more than 100 energy efficiency projects and is now remotely managing more than 100MW load including lakhs of street lamps and tens of thousands of pumps.
# 
# 

# 
# 
# CCMS or the Standalone Street Light Controller is a control panel with comprehensive protection, control and monitoring station for a group of street lights. It includes a Class1.0 metering unit and communicates to the SMART web server with GSM/GPRS connection.
# 

# ## Key Highlights of Street Light controller:
Over / under voltage protection
Over load protection
Short circuit protection
Auto rectification for nuisance MCB Trips
Tolerant to input voltage fluctuations
Surge protection up to 40 KA
Astronomical / Photo cell / Configurable ON/OFF timings
Event notification for faults
# ## Problem Statement:
# We are tasked with predicting the number of units consumed by each street light controller.The data  is received the IoT device which is deployed in the various states in india.

# ## Mapping the real world problem to an ML problem
# 
# Type of Machine Leaning Problem
# 
# Supervised Learning:
# 
# It is a regression problem, for a given data we need to predict the energy consumption of the street light controller

# ## Train and Test Construction
# We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.

# Importing the Necessary Libraries

# In[123]:


import numpy as np
import pandas as pd


# ## Reading the data

# In[ ]:


data=pd.read_csv('Day_Report (VARANASHI) - dayreport.csv')
data


# ## How many Missing values in the dataset?

# In[125]:


#Credit:Prof-Vejey
for i in range(len(data.columns)):
    missing_data = data[data.columns[i]].isna().sum()
    perc = missing_data / len(data) * 100
    print(f'Feature {i+1} >> Missing entries: {missing_data}  |  Percentage: {round(perc, 2)}')


# ## Visual representation of missing values

# In[126]:


plt.figure(figsize=(10,6))
sns.heatmap(data.isna(), cbar=False, cmap='viridis', yticklabels=False)


# We do not have any missing values in this dataset

# ## Handling Categorical data

# In[127]:


#Converting the categorical variable into numerical using lable encoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
xm=data.apply(LabelEncoder().fit_transform)
xm


# ## Splitting the data 

# Input features(Independent Variables)

# In[128]:


X = xm.iloc[:,:22]
X


# Output Features(Dependent Variable)

# In[129]:


y=xm.iloc[:,-3]
y


# # Shape

# In[130]:


X.shape, y.shape


# # Plotting the correlation values for each feature

# In[131]:


xm.corr()


# # Plotting the heat map

# In[233]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#get correlations of each features in dataset
corrmat = xm.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(25,25))
#plot heat map
g=sns.heatmap(xm[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Feature Importance

# In[260]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt


# In[261]:


model = ExtraTreesRegressor()


# In[262]:


model.fit(X,y)


# In[113]:


print(model.feature_importances_)


# In[263]:



#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# # Building the Model

# # 1.Random Forest Regressor

# In[268]:


from sklearn.ensemble import RandomForestRegressor


# In[269]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[270]:


RFR_model = RandomForestRegressor(n_estimators = 100, random_state = 0)
RFR_model.fit(X_train, y_train)


# In[271]:


y_pred_test = RFR_model.predict(X_test)
y_pred_test


# In[294]:


print('Test_Accuracy',RFR_model.score(X_test, y_test))
print('Train_Accuracy:',RFR_model.score(X_train, y_train))


#  Mean Square Error of the RFR model

# In[273]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
print ("RMS Error: ", rmse)

r_squared = RFR_model.score(X_train, y_train)
print ("R_squared is: ", r_squared)


# # 2. Linear Regression

# In[297]:


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=0)


# In[296]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[298]:


Y_pred = lm.predict(X_test)


# In[299]:


lm.coef_


# In[300]:


Test_Accuracy=lm.score(X_test,y_test)
Train_Accuracy=lm.score(X_train,y_train)
print("Test_Accuracy:",Test_Accuracy)
print("Train_Accuracy:",Train_Accuracy)


# Mean Square Error of the LR model

# In[301]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
print ("RMS Error: ", rmse)

r_squared = lm.score(X_train, y_train)
print ("R_squared is: ", r_squared)


# # 3.Gradient Boosting Regressor

# In[284]:


from sklearn.ensemble import GradientBoostingRegressor


# In[285]:


reg = GradientBoostingRegressor(random_state=0)


# In[286]:


reg .fit(X_train, y_train)


# In[287]:


y_pred = reg.predict(X_test)


# In[288]:


print("Accuracy:",reg.score(X_test,y_test))
print("Accuracy:",reg.score(X_train,y_train))


# Mean Square Error of the GBR model

# In[289]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
print ("RMS Error: ", rmse)

r_squared = reg.score(X_train, y_train)
print ("R_squared is: ", r_squared)


# # 5.Conclusion

# In[308]:


print('\n                     Accuracy     Error')
print('                     ----------   --------')
print('Linear Regression   : {:.04}%        {:.04}%'.format( lm.score(X_test, y_test)* 100,                                                  100-(lm.score(X_test, y_test) * 100)))

print('Random Forest       :  {:.04}%        {:.04}% '.format(RFR_model.score(X_test, y_test)* 100,                                                           100-(RFR_model.score(X_test, y_test)* 100)))
print('Gradient Boosting   : {:.04}%        {:.04}% '.format(reg.score(X_test, y_test)* 100,                                                           100-(reg.score(X_test, y_test)* 100)))

