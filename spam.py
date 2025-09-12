import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Basic data preprocessing
df=pd.read_csv('spam.csv')

print(df.info())

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
print(df.info())

df.rename(columns={'v1': 'target','v2': 'text'},inplace=True)
print(df.head())

# handling null values
print(df.isnull().sum())

# handling duplicates values
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.shape)


# EDA

df['target'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("ham and spam distribution")
plt.legend()
plt.show()


nltk.download('punkt_tab')
df['num_chars']= df['text'].apply(len)
df['num_words']= df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sen']= df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


print(df[['num_chars','num_sen','num_words']].describe())

print(df[df['target'] == 'ham'][['num_chars','num_sen','num_words']].describe())

print(df[df['target'] == 'spam'][['num_chars','num_sen','num_words']].describe())

