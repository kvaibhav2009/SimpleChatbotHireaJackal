from bs4 import BeautifulSoup
import requests
import pandas as pd


def createTrainingData():
    link = "https://www.hbs.edu/mba/find-answers/Pages/default.aspx"
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html5lib")

    questions = []
    for question in soup.find_all("dt"):
        questions.append(question.get_text())

    answers = []
    for answer in soup.find_all("dd"):
        answers.append(answer.get_text())

    TrainingData = pd.DataFrame(list(zip(questions, answers)), columns=['Question', 'Answer'])

    return TrainingData




TrainingData=createTrainingData()
