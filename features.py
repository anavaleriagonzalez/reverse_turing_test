from sklearn.feature_extraction.text import TfidfVectorizer
from laserembeddings import Laser
from bert_embedding import BertEmbedding
from sklearn import preprocessing
from tqdm import tqdm

def bow(text, vectorizer, train=True):
    if train ==True:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(text)
    X = vectorizer.transform(text)
    return  X, vectorizer


def Bert(text):
    bert_embedding = BertEmbedding()
    X = bert_embedding(text)
  
    return X


def laserembs(text):
    laser = Laser()
    X = laser.embed_sentences(text,
        lang='en')
    return  X

def encode_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    Y = le.transform(labels)
    
    return Y




