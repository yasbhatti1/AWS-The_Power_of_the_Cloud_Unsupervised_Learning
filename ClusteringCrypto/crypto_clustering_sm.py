#!/usr/bin/env python
# coding: utf-8

# # Clustering Crypto

# In[1]:


# Initial imports
#!pip install altair
import requests
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import altair as alt


# ### Fetching Cryptocurrency Data

# In[2]:


# Use the following endpoint to fetch json data
url = "https://min-api.cryptocompare.com/data/all/coinlist"


# In[3]:


#Create a DataFrame 
# HINT: You will need to use the 'Data' key from the json response, then transpose the DataFrame.

resp = requests.get(url)
response_content = resp.content
data = resp.json()


# In[4]:


# save crypto data as coin_data
coin_data = data['Data']


# In[5]:


# Create a Datafrane and then Transpose datframe (Columns to Rows)
crypto_df = pd.DataFrame(coin_data)
crypto_df = crypto_df.T
crypto_df.head()


# In[6]:


# Select desired columns from above dataframe and save it as 'crypto_df'
crypto_df = crypto_df[['CoinName', 'Algorithm', 'IsTrading', 'ProofType', 
                         'TotalCoinsMined', 'CirculatingSupply']]
crypto_df


# ### Data Preprocessing

# In[7]:


# check the shape (rows and columns) of dataframe
crypto_df.shape


# In[8]:


# Keep only cryptocurrencies that are trading
crypto_df = crypto_df.loc[crypto_df['IsTrading'] == True].copy()
crypto_df.shape


# In[9]:


# Replace Algorithm column empty cells with numpy nan 
crypto_df['Algorithm'].replace('', np.nan, inplace=True)


# In[10]:


# drop na from column 'Algorithm'
crypto_df.dropna(subset=['Algorithm'], inplace=True)


# In[11]:


# Keep only cryptocurrencies with a working algorithm
crypto_df = crypto_df.loc[crypto_df['Algorithm'] != 'N/A' ].copy()
crypto_df.shape


# In[12]:


# Remove the "IsTrading" column
crypto_df.drop(columns=['IsTrading'], inplace = True)


# In[13]:


# check the shape (rows and columns) of dataframe
crypto_df.shape


# In[14]:


# check the count for null values by columns
crypto_df.isnull().sum()


# In[15]:


# Remove rows with cryptocurrencies having no coins mined
crypto_df = crypto_df.loc[crypto_df.TotalCoinsMined > 0].copy()
crypto_df.shape


# In[16]:


# check for any duplicates
print(f"Duplicate entries: {crypto_df.duplicated().sum()}")


# In[17]:


# check the count for null values by columns
crypto_df.isnull().sum()


# In[18]:


# Store the 'CoinName' column in its own DataFrame prior to dropping it from crypto_df. Plus save dataframe 
# in new dataframe named 'shoppings_df' for later usage
coin_df = pd.DataFrame(crypto_df['CoinName'], index=crypto_df.index)
shoppings_df = crypto_df
shoppings_df


# In[19]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm
crypto_df = crypto_df.drop(columns=['CoinName'])
crypto_df.head(10)


# In[20]:


# Create dummy variables for text features only using 'Algorithm' and 'ProofType' columns
crypto_df = pd.get_dummies(crypto_df, columns=['Algorithm', 'ProofType'])
crypto_df


# In[21]:


# Standardize data
crypto_scaled = StandardScaler().fit_transform(crypto_df)
print(crypto_scaled[0:3])


# ### Reducing Dimensions Using PCA

# In[22]:


# Use PCA to reduce dimensions to 3 principal components
pca = PCA(n_components=3)
crypto_pca = pca.fit_transform(crypto_scaled)


# In[23]:


# Create a DataFrame with the principal components data
df_crypto_pca = pd.DataFrame(
    data=crypto_pca, columns=["PC 1", "PC 2", "PC 3"], index=crypto_df.index
)
df_crypto_pca


# ### Clustering Crytocurrencies Using K-Means
# 
# #### Find the Best Value for `k` Using the Elbow Curve

# In[24]:


# Find the best value for k using Elbow Curve
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(crypto_df)
    inertia.append(km.inertia_)


# Create the Elbow Curve
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)


# In[25]:


# Create Elbow chart using altair
alt.Chart(df_elbow).mark_line().encode(
    x='k',
    y='inertia'
).interactive() 


# In[26]:


# Initialize the K-Means model
model = KMeans(n_clusters= 4, random_state=0)

# Fit the model
model.fit(df_crypto_pca)

# Predict clusters
predictions = model.predict(df_crypto_pca)

# Add the predicted class columns
df_crypto_pca["class"] = model.labels_
df_crypto_pca.head()


# In[27]:


# combined crypto data as clustered_df, drop index column, and set index 'Symbol'
clustered_df = pd.concat([shoppings_df, df_crypto_pca], axis=1)
clustered_df.head(10)


# ### Visualizing Results
# 
# #### 2D-Scatter with Clusters

# In[28]:


# Create a 2D-Scatter with the PCA data and the clusters
alt.Chart(clustered_df).mark_circle(size=60).encode(
    x='PC 1',
    y='PC 2',
    color='class',
    tooltip=['CoinName', 'Algorithm', 'TotalCoinsMined', 'CirculatingSupply'],
).interactive()


# #### Table of Tradable Cryptocurrencies
# 

# In[29]:


# for MinMaxscaler usage, we will save totalcoinsmined and CirculatingSupply columns in new dataframe
clustered_df_temp = clustered_df[['TotalCoinsMined', 'CirculatingSupply']]
clustered_df_temp


# In[30]:


# Table with tradable cryptos without principal components, TotalcoinsMined, CirculatingSupply
clustered_df.drop(columns=['TotalCoinsMined', 'CirculatingSupply', 'PC 1', 'PC 2', 'PC 3'], inplace = True)
clustered_df


# #### Scatter Plot with Tradable Cryptocurrencies

# In[31]:


# Use MinMaxScaler on clusterd_df_temp we previously created 
scaler = MinMaxScaler()
tradable_crypto_scaled = scaler.fit_transform(clustered_df_temp)
print(tradable_crypto_scaled[0:10])


# In[32]:


# Create dataframe using tradable_crypto_scaled, Rename columns for scatter plot usage
crypto_scaled_df = pd.DataFrame(tradable_crypto_scaled, index=clustered_df.index)
crypto_scaled_df.rename(columns={0:'TotalCoinsMined', 1: 'TotalCoinSupply'}, inplace = True)
crypto_scaled_df


# In[33]:


# combine scaled dataframe with clustered dataframe using index
tradable_crypto_df = pd.concat([crypto_scaled_df, clustered_df], axis = 1)
tradable_crypto_df


# In[34]:


# Create a 2D-Scatter 
alt.Chart(tradable_crypto_df).mark_circle(size=60).encode(
    x='TotalCoinsMined',
    y='TotalCoinSupply',
    color='class',
    tooltip=['CoinName', 'Algorithm', 'TotalCoinsMined', 'TotalCoinSupply']
    
).interactive()  


# In[ ]:




