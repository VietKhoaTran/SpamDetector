import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
messages = pd.read_csv('Projects/data/SMSSpamCollection', sep = '\t', names = ['label', 'message'])
# sep = '/t': specifies the seperator used in the file, which is tab in this case

import re
import nltk
# Data clearing and preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
result = re.sub('[^a-zA-z]', ' ', messages['message'][1]) # ^ for negating the character set
def clearing(messages):
    review = re.sub('[^a-zA-Z]', ' ', messages)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review
corpus = [clearing(messages['message'][i]) for i in range(len(messages))]

#Creating bag of words:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus).toarray()    
y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
Naive_bayes = MultinomialNB()
Naive_bayes.fit(x_train, y_train)
y_pred = Naive_bayes.predict(x_test)
new_message = 'Had your contract mobile 11 Mnths? Latest Motorola, Nokia etc. all FREE! Double Mins & Text on Orange tariffs. TEXT YES for callback, no to remove from records.'
new_message = clearing(new_message)
new_message = vectorizer.transform([new_message]).toarray()
predicted_category = Naive_bayes.predict(new_message)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Predicted category of '{new_message}' is {'spam' if predicted_category[0] == 1 else 'ham'}")