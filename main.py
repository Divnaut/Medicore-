import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import json
from symptoms import Symptoms
from sklearn.metrics.pairwise import cosine_similarity
import heapq

#creating the app object:
app=FastAPI()
dataset=pd.read_csv(r'Training.csv')
dataset=dataset.drop(columns=['fluid_overload.1'])
grouped_df = dataset.drop(columns=['prognosis']).groupby(dataset['prognosis']).mean()
description=pd.read_csv(r'C:\Users\div\OneDrive\Documents\GitHub\Medicore-\description.csv')
diets=pd.read_csv(r'C:\Users\div\OneDrive\Documents\GitHub\Medicore-\diets.csv')
medications=pd.read_csv(r'C:\Users\div\OneDrive\Documents\GitHub\Medicore-\medications.csv')
workout=pd.read_csv(r'C:\Users\div\OneDrive\Documents\GitHub\Medicore-\workout_df.csv')


@app.post('/predict')
def predict_disease(data: Symptoms):
    data=data.dict()
    arr=[]
    top_5_heap=[]
    #creating a list of symptoms with 0's and 1's at appropriate positions
    for value in data.values():
        arr.append(value)
    recieved_symptoms=pd.Series(arr, index=dataset.columns[:-1])  
    for disease, row in grouped_df.iterrows():
        similarity = cosine_similarity([row.values], [recieved_symptoms.values])[0][0]
        if len(top_5_heap) < 5:
            heapq.heappush(top_5_heap, (similarity, disease))
        else:
            heapq.heappushpop(top_5_heap, (similarity, disease)) 
    top_5_sorted = sorted(top_5_heap, key=lambda x: -x[0])
    #creating a dictionary of top 5 diseases and similarity index and returning it :
    prediction={}
    for similarity, disease in top_5_sorted:

        prediction[f'{disease}']={}
    return prediction



