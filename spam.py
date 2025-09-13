import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


nltk.download('punkt_tab')
nltk.download('stopwords')


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

# df['target'].value_counts().plot.pie(autopct='%1.2f%%')
# plt.title("ham and spam distribution")
# plt.legend()
# plt.show()



df['num_chars']= df['text'].apply(len)
df['num_words']= df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sen']= df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

print(df[['num_chars','num_sen','num_words']].describe())

print(df[df['target'] == 'ham'][['num_chars','num_sen','num_words']].describe())

print(df[df['target'] == 'spam'][['num_chars','num_sen','num_words']].describe())


# sns.pairplot(df,hue='target')
# plt.show()


# Encoding data for ham and spma so that the machine can understand
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df[['target']])


num_data=df[['target','num_chars','num_sen','num_words']]
sns.heatmap(num_data.corr(),annot=True)
plt.show()


# ***********Data preprocessing***********
# lower case
# Tokenization
# Removing special chars
# Removing stop words and punctuation
# stemming

def text_transform(text):
    text= text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    ps=PorterStemmer()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


