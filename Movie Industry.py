#!/usr/bin/env python
# coding: utf-8

# In[224]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[225]:


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] =(12,8)


# In[226]:


df = pd.read_csv('/Users/aliariyan/Desktop/Apply/Portfolio/Correlation in Python/movies.csv')


# In[227]:


df.head()


# In[228]:


# Data Types for our columns

df.dtypes


# In[229]:


df.describe()


# In[230]:


#change data type of columns

#df['budget'] = df['budget'].astype('Int64')
#df['gross'] = df['gross'].astype('Int64')


# In[231]:


# Let's see if there is any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[232]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[233]:


# Data Types for our columns

df.dtypes


# In[234]:


#change data type of columns

df['budget'] = df['budget'].astype('float')
df['gross'] = df['gross'].astype('float')


# In[235]:


#creat correct year column
df['yearcorrect'] =  df['released'].astype('string').str.split(',').str[1].str[1:5]
df.dtypes


# In[236]:


df = df.sort_values(by = ['gross'], inplace = False, ascending = False)


# In[237]:


pd.set_option('display.max_rows', None)


# In[238]:


# Drop any duplicates

df.drop_duplicates()


# In[239]:


# Drop NA Information

df.dropna(inplace=True)
len(df)


# In[240]:


# budget high correlation
# Company high correlation


# In[241]:


# Scater plot with budget vs gross

df.dropna(inplace=True)

plt.scatter(x = df['budget'], y = df['gross'])

plt.title('Budget vs Gross Earning')

plt.xlabel('Gross Earning')

plt.ylabel('Budget for film')

plt.show()


# In[242]:


df.head()


# In[244]:


# Plot budget vs gross using seaborn


sns.regplot(data = df, x = 'budget', y = 'gross', line_kws = {"color" : "blue"})


# In[247]:


# Let's start looking at correlation

df.corr(method = 'pearson') #pearson, kendall, spearman


# In[249]:


correlation_matrix = df.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[250]:


# Looks at Company

df.head()


# In[258]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype =='object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized.head()


# In[253]:


correlation_matrix = df_numerized.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[254]:


df_numerized.corr()


# In[255]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[256]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[257]:


sorted_pairs[(sorted_pairs) > 0.5]


# In[ ]:


# Votes and Budget have the highest correlation to gross earning

