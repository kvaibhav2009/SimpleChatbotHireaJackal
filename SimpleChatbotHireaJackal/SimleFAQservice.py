from flask import Flask,request
import json
import Classifier as classifier
import pandas as pd

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'



@app.route('/getAnswer',methods = ['POST'])
def getAnswer():
    data = request.data
    dataDict = json.loads(data)

    query = dataDict['query']
    #topn = dataDict['topn']

    question, answer,result=classifier.evaluate(query)
    # result=pd.Dataframe(result)
    # result=(result.sort_values(['Score'])[:3]).reset_index(drop=True)
    # result.Answer

    jsondata = {
        'request-type': 'ingest_data',
        'response': {
            "query": "reviews_file_path",
            "answer 1": [str(result.Answer[0]),str((1-float(result.Score[0]))*100)],
            "answer 2": [str(result.Answer[1]),str((1-float(result.Score[1]))*100)],
            "answer 3": [str(result.Answer[2]),str((1-float(result.Score[2]))*100)]

        }
    }
    jsonreturn = json.dumps(jsondata)
    print(jsonreturn)
    return jsonreturn

if __name__ == '__main__':
    app.run()
