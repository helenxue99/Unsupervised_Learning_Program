#!/usr/bin/env python
# coding: utf-8

# # BBC News Article Recommender  System
#  

# ## 1. Purpose
# 
# As we learned and do a lot of exercise on this Unsuperisor learning course, like recommender  system and News Article category. But I am thinking about in the real world, while we surf on internet and read a news article, the system cannot get our personal privacy data, so mostly for a news system they need a recommender system on content base. 
# 
# On Week4 we did the BBC News catgory programming, is there a way we can do a recommender system for BBC News Article

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
 
import os


# ## 2. Load BBC Articles Data

# In[2]:


data_train = pd.read_csv('E:\\University of Colorado Boulder\\12- Unsupevised Algorithms in Machine Learning\\Week4\\learn-ai-bbc\\BBC News Train.csv', delimiter=',')


# ## 3. Extracting word features and show Exploratory Data Analysis (EDA)

# In[3]:


print(data_train.info())
print(data_train.head())


# In[4]:


print(data_train.describe(include = 'object'))


# In[5]:


print("Total observations ", len(data_train))
print("Total Count of Unique Article IDs ", len(data_train['ArticleId'].unique()))


# In[6]:


plt.hist(data_train['Category'])
plt.title("Histogram of Training Data Categories")
print("Tech the smallest category makes up this percentage:", round(len(data_train[data_train['Category'] == 'tech'])/ len(data_train),3))
print("Sport the largest category makes up this percentage:", round(len(data_train[data_train['Category'] == 'sport'])/ len(data_train),3))
categories = data_train['Category'].unique()


# In[7]:


a = []
for txt in data_train['Text']:
    a.append(len(txt.split()))

print(len(txt.split()) )
plt.hist(a)
plt.title("Word Count")
print("Smallest Article ", min(a))
print("Largest Article " , max(a))


# In[8]:


data_train.isna().sum()


# In[9]:


data_train['Category'].value_counts()


# In[10]:


target_category = data_train['Category'].unique()
print(target_category)


# In[11]:


news_cat = data_train['Category'].value_counts()

plt.figure(figsize=(10,6))
my_colors = ['r','g','c','m','b']
news_cat.plot(kind='bar', color=my_colors)
plt.grid()
plt.xlabel("News Categories")
plt.ylabel("Datapoints Per Category")
plt.title("Distribution of Datapoints Per Category")
plt.show()


# # 4. Clean Data

# In[12]:


print(data_train.isna().sum())
# data_train =data_train.fillna(0)       
# data_train[[  'website','title' ,'content']] = data_train[[ 'website','title' ,'content']].fillna(value=0)
data_train[[   'Text' ,'Category']] = data_train[[ 'Text' ,'Category']].fillna(value='')
data_train.isna().sum()


# In[13]:


#lowercase
data_train['Text'] = data_train.Text.apply(lambda x: x.lower())
data_train['Category'] = data_train['Category'].astype(str)
data_train['Category'] = data_train.Category.apply(lambda x: x.lower())
print(data_train['Text']) 


# In[14]:


import string
table = str.maketrans("", "", string.punctuation)
def remove_punc(text):
    return text.translate(table)


# In[15]:


data_train['Text'] = data_train.Text.apply(lambda x: remove_punc(x))
data_train['Category'] = data_train.Category.apply(lambda x: remove_punc(x))
 
data_train.head(2)


# In[16]:


#Remove stopwords

from nltk.corpus import stopwords

stop = set(stopwords.words("english"))

def rem_stop(text):
    word_list = [word for word in text.split() if word not in stop]
    return " ".join(word_list)


# In[17]:


data_train['Text'] = data_train.Text.apply(lambda x: rem_stop(x))
data_train['Category'] = data_train.Category.apply(lambda x: rem_stop(x))
 
data_train.head(2)


# In[18]:


a = []
for txt in data_train['Text']:
    a.append(len(txt.split()))

print(len(txt.split()) )
plt.hist(a)
plt.title("Word Count")
print("Smallest Article ", min(a))
print("Largest Article " , max(a))


# In[19]:


# # Convert a collection of raw documents to a matrix of TF-IDF features
# from sklearn.decomposition import NMF 
 
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidvec = TfidfVectorizer(min_df = 2,
#                           max_df = 0.95,
#                           norm = 'l2',
#                           stop_words = 'english')
# tfidvec_train = tfidvec.fit_transform(data_train['Text'] )


# ## 5. Bag of Words

# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer()
bowmatrix = bow.fit_transform(data_train['Text']).toarray()


# In[21]:


print(bowmatrix.shape)


# ## 6. Recommender (Content based)

# ## 6.1 cosine_similarity

# In[22]:


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
nn = NearestNeighbors()
score = cosine_similarity(bowmatrix)


# In[23]:


print(score)


# In[24]:


def Neighbor_by_cosine(article):
    row_num = data_train[data_train['ArticleId'] == article].index.values[0] #getting the index of the article
    similarity_score = list(enumerate(score[row_num])) #similar articles
    sorted_score = sorted(similarity_score, key=lambda x:x[1], reverse= True)[1:6] #sorting similar articles and returning the first 5
    print('Current Article:',data_train.iloc[row_num])
    i = 0
    print('Cosine Recommend Article:')
    recommendations =[]
    for item in sorted_score:
 
        print(data_train.iloc[data_train.index == item[0]])
        article_id = data_train[data_train.index == item[0]]["ArticleId"].values[0] #getting the article title
        category = data_train[data_train.index == item[0]]["Category"].values[0] #getting the article title
        Text = data_train[data_train.index == item[0]]["Text"].values[0] #getting the article title
        recommendations.append([i+1,article_id, category]) 
 
        i = i + 1
    return recommendations #returns the 5 nearest article titles


# ## 6.2 Jaccard_similarity

# In[25]:


#  Use Jaccard to get the similar article
from sklearn.metrics.pairwise import pairwise_distances 
jaccard = pairwise_distances(bowmatrix, bowmatrix, metric="jaccard", n_jobs=-1)
jaccard = 1 - jaccard


# In[26]:


def Neighbor_by_jaccard(article):
    row_num = data_train[data_train['ArticleId'] == article].index.values[0] #getting the index of the article
    similarity_score = list(enumerate(jaccard[row_num])) #similar articles
    sorted_score = sorted(similarity_score, key=lambda x:x[1], reverse= True)[1:6] #sorting similar articles and returning the first 5
    print('Current Article:',data_train.iloc[row_num])
    i = 0
    print('Jaccard Recommend Article:')
    recommendations =[]
    for item in sorted_score:
 
        print(data_train.iloc[data_train.index == item[0]])
        article_id = data_train[data_train.index == item[0]]["ArticleId"].values[0] #getting the article title
        category = data_train[data_train.index == item[0]]["Category"].values[0] #getting the article title
        Text = data_train[data_train.index == item[0]]["Text"].values[0] #getting the article title
       
        recommendations.append([i+1,article_id, category]) 
 
        i = i + 1
    return recommendations #returns the 5 nearest article titles


# ## 7. Verify the Recommender  Result

# The BBC Article have 5 categories, to verify the recommand articles result on content base, we have to check if the  recommend articles are really similar or related to the original article.
# 
# I think we can verify it from the following two items:
# 
#     1. The recommend article should belong to the same category
#     2. The recommend article should discuss the same or similar topic, The BBC article dataset didn't provide us the title, we have to find the topic from content
#     
# So we will verify all five categories:
# 
#     Business
#     Tech
#     entertainment
#     politics
#     sport
#     

# ### 7.1 Verify Business Category

# In[27]:


Neighbor_by_cosine(1833)
Neighbor_by_jaccard(1833)


# ### 7.2 Verify Tech Category

# In[28]:


Neighbor_by_cosine(1976)
Neighbor_by_jaccard(1976)


# ### 7.3 Verify entertainment Category

# In[29]:



Neighbor_by_cosine(177)
Neighbor_by_jaccard(177)


# ### 7.4 Verify politics Category

# In[30]:


Neighbor_by_cosine(2100)
Neighbor_by_jaccard(2100)


# ### 7.5 Verify sport Category

# In[31]:


Neighbor_by_cosine(1026)
Neighbor_by_jaccard(1026)


# ## 8. Conclusion

# Recommend system works for using  unsuperivsor learning to get the content similar article
# 
# Cosine Method is better than Jaccard distance Method

# In[ ]:




