import nltk
import re# re libray will use for regular expression 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from  nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize




paragraph='''Natural language processing (NLP) refers to the branch 
of computer science—and more specifically, the branch of artificial 
intelligence or AI—concerned with giving computers the ability to 
understand text and spoken words in much the same way human beings can.

NLP combines computational linguistics—rule-based modeling of human 
language—with statistical, machine learning, and deep learning models. 
Together, these technologies enable computers to process human language 
in the form of text or voice data and to ‘understand’ its full meaning, 
complete with the speaker or writer’s intent and sentiment.'''


# Cleaning the texts
ps = PorterStemmer()
wnt=WordNetLemmatizer()
sen =sent_tokenize(paragraph)


corpus = []

# Create the empty list name as corpus becuase after cleaned the data corpus will store this clean data
for i in range(len(sen)):
    arrange = re.sub('[^a-zA-Z]', ' ', sen[i])
    arrange = arrange.lower()
    arrange = arrange.split()
    arrange = [wnt.lemmatize(word) for word in arrange if not word in set(stopwords.words('english'))]   
    arrange = ' '.join(arrange)
    corpus.append(arrange)

# Creating the Bag of Words model 

# Also we called as document matrix 


from sklearn.feature_extraction.text import CountVectorizer
cv_1 = CountVectorizer()
X_1 = cv_1.fit_transform(corpus).toarray()
 

