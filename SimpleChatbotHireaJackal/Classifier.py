import ScrapeForDataset as scraper
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import wordVectorRepresentation as wordvec
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import spatial
import xlrd
import math
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
import re
from scipy.spatial import distance
import gensim.downloader as api
from gensim.models import KeyedVectors
from pathlib import Path
import os
import sys

sys.path.append(os.path.realpath(os.getcwd()))
stopword = set(stopwords.words('english'))


def CheckModel():
    print("initialize")
    my_file = Path("model1/glove-wiki-gigaword-200")
    if my_file.is_file():
        print("File is present")
        filepath = os.getcwd() + "/model/glove-wiki-gigaword-200"
        model = KeyedVectors.load(filepath)
    else:
        print("Downloading file")
        info = api.info()
        model = api.load("glove-wiki-gigaword-200")
        filepath=os.getcwd() + "/model1/glove-wiki-gigaword-200"
        model.save(filepath)
        print("Downloading complete")


    return

def getIDF(TrainingSet):
    whWords = set(['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how'])
    stopwords = stopword - whWords
    vectorizer = TfidfVectorizer(min_df=1,stop_words=stopwords, sublinear_tf=True)
    lemmatizer = WordNetLemmatizer()
    X=TrainingSet.Question
    X.append(TrainingSet.Answer)
    vectorizer.fit_transform(X)
    tfvalues=[lemmatizer.lemmatize(x) for x in list(vectorizer.vocabulary_.keys())]# vectorizer.vocabulary_
    idfvalues = vectorizer.idf_
    IDFset=dict(zip(tfvalues,idfvalues))

    return IDFset


TrainingSet=scraper.createTrainingData()

IDFset=getIDF(TrainingSet)


Question_vectors=wordvec.questiontoVector(TrainingSet,IDFset)



# score = spatial.distance.cosine(query_vector, vec)
#utterence="Who should recommendations come from"

def getAnswer(utterence):

    y=wordvec.vectorize_query(utterence,IDFset)
    scores=[]
    for i in range(Question_vectors.__len__()):
        x = np.array(Question_vectors[i])
        score = spatial.distance.euclidean(x, y)
        if math.isnan(score):
            scores.append(1)
        else:
            scores.append(score)


    index=scores.index(np.min(scores))
    print("Index",index)
    answer=TrainingSet.Answer[index]
    question=TrainingSet.Question[index]

    return question,answer


def evaluateAnswers():
    questions=pd.read_excel("Test Questions.xlsx")
    Testquestion = []
    Testanswer = []
    ClassifiedQuestion=[]
    for i in range(questions.__len__()):
        utterence = questions.values[i].item()
        question,answer = getAnswer(utterence)
        ClassifiedQuestion.append(question)
        Testanswer.append(answer)
        Testquestion.append(utterence)

    TestSet = pd.DataFrame(list(zip(Testquestion,ClassifiedQuestion, Testanswer)), columns=['Question','ClassifiedQuestion', 'Answer'])
    TestSet.to_excel("TestSet20.xlsx")

    return TestSet
def Data_Cleaner(sentence_text):
    sentence_text = re.sub('[/?]+', ' ', sentence_text)
    sentence_text = sentence_text.replace("HBS", "harvard business school")
    sentence_text=re.sub('[^A-Za-z0-9 ]+', '', sentence_text)
    lemmatize = True
    stem = False
    remove_stopwords = True

    whWords = set(['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how'])
    stopwords = stopword - whWords
    stops = stopword - whWords

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
filepath=os.getcwd()+"/model/glove-wiki-gigaword-200"

model=KeyedVectors.load(filepath)
#model=Word2Vec.load("model/SoftModel_w2v2")

def evaluate(utterance):
    query = wordvec.Data_Cleaner(utterance)
    scores = []
    for row in list(zip(TrainingSet.Question, TrainingSet.Answer)):                    # for row in TrainingSet.Question:
        sentence = row[0]  + " " + row[1]
        x =wordvec.Data_Cleaner(sentence)
        score = model.wmdistance(x, query)
        scores.append(score)

    index = scores.index(np.min(scores))

    question = TrainingSet.Question[index]
    answer = TrainingSet.Answer[index]

    result = pd.DataFrame(list(zip(TrainingSet.Question,TrainingSet.Answer, scores)),
                           columns=['Question', 'Answer', 'Score'])

    result=result.sort_values(['Score'])[:3]
    result=result.reset_index(drop=True)
    return  question,answer,result

def evaluateCorrectAnswers():
    questions=pd.read_excel("Test Questions.xlsx")
    Testquestion = []
    Testanswer = []
    ClassifiedQuestion=[]
    for i in range(questions.__len__()):
        utterence = questions.values[i].item()
        question,answer = evaluate(utterence)
        ClassifiedQuestion.append(question)
        Testanswer.append(answer)
        Testquestion.append(utterence)

    TestSet = pd.DataFrame(list(zip(Testquestion,ClassifiedQuestion, Testanswer)), columns=['Question','ClassifiedQuestion', 'Answer'])
    TestSet.to_excel("TestSet49.xlsx")

    return TestSet

def AnalyzeResultforQuestion(utterence,no):

    y=wordvec.vectorize_query(utterence,IDFset)
    scores=[]
    query=Data_Cleaner(utterence)
    processedQuestion=[]
    for i in range(Question_vectors.__len__()):
        x=Data_Cleaner(TrainingSet.Question[i]) #x = np.array(Question_vectors[i])
        score=model.wmdistance(query,x) #score = spatial.distance.cosine(x, y)
        processedQuestion.append(x)
        if math.isnan(score):
            scores.append(1)
        else:
            scores.append(score)

    TestSet = pd.DataFrame(list(zip(TrainingSet.Question,processedQuestion, scores)),columns=['Question','Processed Question', 'Score'])
    utterence=re.sub('[^A-Za-z0-9 ]+', '', utterence)
    excel_name=str('_'.join(utterence.split())) +".xlsx"
    excel_name="TestSet"+no+".xlsx"
    TestSet.to_excel(excel_name)

    index=scores.index(np.min(scores))
    print("Index",index)
    answer=TrainingSet.Answer[index]
    question=TrainingSet.Question[index]

    return query,question,answer,np.min(scores)


def AnalyzeResultforQuestionWithIDF(utterence,no):

    y=wordvec.vectorize_query(utterence,IDFset)
    scores=[]
    query=Data_Cleaner(utterence)
    processedQuestion=[]
    for i in range(Question_vectors.__len__()):
        x = np.array(Question_vectors[i])
        score = spatial.distance.cosine(x, y)
        processedQuestion.append(Data_Cleaner(TrainingSet.Question[i]))
        if math.isnan(score):
            scores.append(1)
        else:
            scores.append(score)

    TestSet = pd.DataFrame(list(zip(TrainingSet.Question,processedQuestion, scores)),columns=['Question','Processed Question', 'Score'])
    utterence=re.sub('[^A-Za-z0-9 ]+', '', utterence)
    excel_name=str('_'.join(utterence.split())) +".xlsx"
    excel_name="TestSet"+no+".xlsx"
    TestSet.to_excel(excel_name)

    index=scores.index(np.min(scores))
    print("Index",index)
    answer=TrainingSet.Answer[index]
    question=TrainingSet.Question[index]

    return query,question,answer,np.min(scores)

def evaluateBatchQuestion():
    questions=pd.read_excel("Test Questions.xlsx")
    Testquestion = []
    Testanswer = []
    ClassifiedQuestion=[]
    for i in range(questions.__len__()):
        utterence = questions.values[i].item()
        question,answer = getAnswer(utterence)
        ClassifiedQuestion.append(question)
        Testanswer.append(answer)
        Testquestion.append(utterence)

    TestSet = pd.DataFrame(list(zip(Testquestion,ClassifiedQuestion, Testanswer)), columns=['Question','ClassifiedQuestion', 'Answer'])
    TestSet.to_excel("TestSet51.xlsx")

    return TestSet


#evaluateCorrectAnswers()


#evaluateAnswers()

# stop_words = stopwords.words('english')
# utterance="how much  of  experience you  to  in candidates?"
# q2="What percentage of students lives on campus?"
# p1 = [w for w in utterance if w not in stop_words]
# p2= [w for w in q2.lower().split() if w not in stop_words]
#
# model.wmdistance(p1,p2)
# def evaluate(utterence):
#     query = [w for w in utterance.lower().split() if w not in stop_words]
#     scores = []
#     for row in TrainingSet.Question:
#         x = [w for w in row.lower().split() if w not in stop_words]
#         score = model.wmdistance(x, query)
#         scores.append(score)
#
#     index = scores.index(np.min(scores))
#     question = TrainingSet.Question[index]
#     answer = TrainingSet.Answer[index]
#utterence="How much work experience do I need?"
#AnalyzeResultforQuestion(utterence)
#     return question, answer
#AnalyzeResultforQuestionWithIDF(utterence,'38')
    #utterance="How much work experience do I need?"
# utterance="As a student, will I have the opportunity to network with alumni through organized events?"
# utterance="How can I reach out to  alumni for guidance?"
# #query,question,answer,score=AnalyzeResultforQuestionWithIDF(utterance,'47')
#
# print("Output")
# print("\n Query",query)
# print("\n Question",question)
# print("\n Answer",answer)
# print("\n Score",score)
# print("------BatchTesting------")
# #evaluateBatchQuestion()
# #evaluateCorrectAnswers()
# print("------BatchTesting Ends------")

CheckModel()