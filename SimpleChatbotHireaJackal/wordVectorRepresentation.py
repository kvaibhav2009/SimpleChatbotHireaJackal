from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
import re



stops = set(stopwords.words("english"))
whWords=set(['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how'])
stops=stops-whWords

def Data_Cleaner(sentence_text):
    sentence_text=re.sub('[/?]+',' ',sentence_text)
    sentence_text = sentence_text.replace("HBS", "harvard business school")
    sentence_text=re.sub('[^A-Za-z0-9 ]+', '', sentence_text)
    lemmatize = True
    stem = False
    remove_stopwords = True
    stops = set(stopwords.words("english"))
    whWords = set(['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how'])
    stops = stops - whWords

    sentence_text =sentence_text.replace("hbs","harvard business school")

    words = sentence_text.lower().split()
    # Optional stemmer
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]

    # Optionally remove stop words (false by default)
    if remove_stopwords:
        words = [w for w in words if not w in stops]

    return words


#model=Word2Vec.load("/Users/vaibhavkotwal/PycharmProjects/ConversationalBankingClassifier/updatedInsurance_word2vec_v3_18650_tri1")  # "model_W2V")
#model=Word2Vec.load("model/SoftModel_w2v2")
#model=Word2Vec.load("model/glove-wiki-gigaword-200")
model=KeyedVectors.load("model/glove-wiki-gigaword-200")
def vectorize_question(utterence,IDFset):
    vec = np.zeros(model.wv.syn0.shape[1]).reshape((1, model.wv.syn0.shape[1]))
    words = Data_Cleaner(utterence)
    count = 0
    for word in words:
        try:
            if word in list(IDFset.keys()):
                idf=IDFset[word]
            else:
                idf=3
            #idf=1
            vec += ((model.wv[word]))*idf
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count

    return vec


def questiontoVector(TrainingSet,IDFset):
    Question_vectors=[]
    for row in list(zip(TrainingSet.Question,TrainingSet.Answer)):
        sentence=row[0]#+" "+row[1]
        sentence_vec=vectorize_question(sentence,IDFset)
        Question_vectors.append(list(sentence_vec))


    return Question_vectors



def vectorize_query(utterence,IDFset):
    vec = np.zeros(model.wv.syn0.shape[1]).reshape((1, model.wv.syn0.shape[1]))
    words = Data_Cleaner(utterence)
    mean=np.mean(list(IDFset.values()))

    count = 0
    for word in words:
        try:
            if word in list(IDFset.keys()):
                idf=IDFset[word]
            else:
                idf=3
            #idf=1
            vec += ((model.wv[word]))*(idf)
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count

    return vec


# vector=[]
# for x in IDFset.keys():
#     if x in model.wv.vocab:
#         vector.append(list(model.wv[x]))
#     else:
#         vector.append(list(np.zeros(model.wv.syn0.shape[1]).reshape((1, model.wv.syn0.shape[1]))))
#
#
# np.mean(list(IDFset.values()))
#
# np.std(list(IDFset.values()))
# Out[6]: 0.3842367710172565
# np.argmin(list(IDFset.values()))
# Out[7]: 155
# np.min(list(IDFset.values()))
# Out[8]: 2.55814461804655
# np.max(list(IDFset.values()))
# Out[9]: 4.637586159726386
# np.mean(list(IDFset.values()))
# Out[10]: 4.439450365956451

# def questiontoVector(TrainingSet,IDFset):
#     Question_vectors=[]
#     for sentence in TrainingSet.Question:
#         sentence_vec=vectorize_question(sentence,IDFset)
#         Question_vectors.append(list(sentence_vec))
#
#
#     return Question_vectors


# import gensim.downloader as api
#
# info = api.info()  # show info about available models/datasets
# model = api.load("glove-twitter-25")