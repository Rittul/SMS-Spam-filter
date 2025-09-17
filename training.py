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

df['target'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("ham and spam distribution")
plt.legend()
plt.show()



df['num_chars']= df['text'].apply(len)
df['num_words']= df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sen']= df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

print(df[['num_chars','num_sen','num_words']].describe())

print(df[df['target'] == 'ham'][['num_chars','num_sen','num_words']].describe())

print(df[df['target'] == 'spam'][['num_chars','num_sen','num_words']].describe())


sns.pairplot(df,hue='target')
plt.show()


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
    text= text.lower()    #for lower case
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:               #removing special chars
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:       #removing stop words which help in sentence formation only only in on etc.
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    ps=PorterStemmer()
    
    for i in text:           #making it stemming like dancing danced as dance only
        y.append(ps.stem(i))
            
    return " ".join(y)


df['transformed_text']= df['text'].apply(text_transform)


# MODEL BIUILDING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB,MultinomialNB
from sklearn.metrics import confusion_matrix,precision_score,accuracy_score

tfidf=TfidfVectorizer()

X=tfidf.fit_transform(df['transformed_text']).toarray()
y=df['target'].values


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=2)

gnb=GaussianNB()
bnb=BernoulliNB()
mnb=MultinomialNB()



print("\nmodel 1: ")
gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print("model accuracy",gnb.score(X_test,y_test))
print("model precission score: ",precision_score(y_test,y_pred1))
print("confusion matrix: ",confusion_matrix(y_test,y_pred1))


print("\nmodel 2: ")
bnb.fit(X_train,y_train)    #this model gives the best score out of every one here
y_pred2=bnb.predict(X_test)
print("model accuracy",bnb.score(X_test,y_test))
print("model precission score: ",precision_score(y_test,y_pred2))
print("confusion matrix: ",confusion_matrix(y_test,y_pred2))


print("\nmodel 3: ")
mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print("model accuracy",mnb.score(X_test,y_test))
print("model precission score: ",precision_score(y_test,y_pred2))
print("confusion matrix: ",confusion_matrix(y_test,y_pred2))



import pickle

with open("spam_model.pkl","wb") as f:
    pickle.dump(bnb,f)
    
with open("vectorizer.pkl","wb") as f:
    pickle.dump(tfidf,f)
    



# model 1:
# model accuracy 0.8762088974854932
# model precission score:  0.5231481481481481
# confusion matrix:  [[793 103]
#  [ 25 113]]

# model 2:
# model accuracy 0.9700193423597679
# model precission score:  0.9734513274336283
# confusion matrix:  [[893   3]
#  [ 28 110]]

# model 3:
# model accuracy 0.9593810444874274
# model precission score:  1.0
# confusion matrix:  [[896   0]
#  [ 42  96]]


