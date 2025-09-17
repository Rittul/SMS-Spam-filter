import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

with open("spam_model.pkl", "rb") as f:
    bnb = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)



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


# ps=PorterStemmer()

messege=input("Enter the message 😊: ")

tranformed_message=text_transform(messege)

input_message=tfidf.transform([tranformed_message]).toarray()

prediction=bnb.predict(input_message)

if prediction[0]==1:
    print("SPAM")
else:
    print("NOT SPAM")