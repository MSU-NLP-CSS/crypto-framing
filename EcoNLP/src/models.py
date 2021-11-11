from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas 
import string
import re
from nltk.stem import SnowballStemmer

def naive_bayes( processed_features, labels ):
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=42)
    
    text_classifier = GaussianNB()
    text_classifier.fit(X_train, y_train)
   
    predictions = text_classifier.predict(X_test)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))

    return text_classifier


def random_forest( processed_features, labels ):
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=42)
    
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
   
    predictions = text_classifier.predict(X_test)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))

    return text_classifier


def drop_feature( dataframe, features ):
    for feature in features:
        dataframe.drop( columns = [feature], inplace = True )
    return dataframe

### Text Normalizing function. Part of the following function was taken from this link. 
def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)
    ## Convert words to lower case and split them
    text = text.lower().split()
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r'\W', ' ',text)
    # remove all single characters
    text= re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Converting to Lowercase
    text = text.lower()
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text


def pre_process_csv( csv_path ):
    df = pandas.read_csv( csv_path )
    df['category'] = df['category'].map(lambda x: 1 if x=='media' else( 2 if x=='influencers' else 3 ))
    print( df.head(10) )
    # apply the above function to df['text']
    df['tweet'] = df['tweet'].map(lambda x: clean_text(x))
    return df
