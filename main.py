import numpy as np
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
from pydantic import BaseModel

app = FastAPI()

filename= 'mental_health_model.pkl'
model = joblib.load(filename)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeatureData(BaseModel):
    feeling_nervous: int
    panic: int
    breathing_rapidly: int
    sweating: int
    trouble_in_concentration: int
    having_trouble_in_sleeping: int
    having_trouble_with_work: int
    hopelessness: int
    anger: int
    over_react: int
    change_in_eating: int
    suicidal_thought: int
    feeling_tired: int
    close_friend: int
    social_media_addiction: int
    weight_gain: int
    material_possessions: int
    introvert: int
    popping_up_stressful_memory: int
    having_nightmares: int
    avoids_people_or_activities: int
    feeling_negative: int
    trouble_concentrating: int
    blamming_yourself: int


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get('/')
def index():
    return {'message': 'Hello, World'}


# Endpoint to receive feature data and return predictions
@app.post("/predict/")
def predict(features: FeatureData):
    data = features.dict()
    feeling_nervous_value = data['feeling_nervous']
    panic_value = data['panic']
    breathing_rapidly_value= data['breathing_rapidly']
    sweating_value = data['sweating']
    trouble_in_concentration_value = data['trouble_in_concentration']
    having_trouble_in_sleeping_value = data['having_trouble_in_sleeping']
    having_trouble_with_work_value = data['having_trouble_with_work']
    hopelessness_value = data['hopelessness']
    anger_value = data['anger']
    over_react_value = data['over_react']
    change_in_eating_value = data['change_in_eating']
    suicidal_thought_value = data['suicidal_thought']
    feeling_tired_value = data['feeling_tired']
    close_friend_value= data['close_friend']
    social_media_addiction_value= data['social_media_addiction']
    weight_gain_value= data['weight_gain']
    material_possessions_value = data['material_possessions']
    introvert_value = data['introvert']
    popping_up_stressful_memory_value = data['popping_up_stressful_memory']
    having_nightmares_value = data['having_nightmares']
    avoids_people_or_activities_value = data['avoids_people_or_activities']
    feeling_negative_value = data['feeling_negative']
    trouble_concentrating_value = data['trouble_concentrating']
    blamming_yourself_value = data['blamming_yourself']

    data = {
        "feeling_nervous": [feeling_nervous_value],
        "panic": [panic_value],
        "breathing_rapidly": [breathing_rapidly_value],
        "sweating": [sweating_value],
        "trouble_in_concentration": [trouble_in_concentration_value],
        "having_trouble_in_sleeping": [having_trouble_in_sleeping_value],
        "having_trouble_with_work": [having_trouble_with_work_value],
        "hopelessness": [hopelessness_value],
        "anger": [anger_value],
        "over_react": [over_react_value],
        "change_in_eating": [change_in_eating_value],
        "suicidal_thought": [suicidal_thought_value],
        "feeling_tired": [feeling_tired_value],
        "close_friend": [close_friend_value],
        "social_media_addiction": [social_media_addiction_value],
        "weight_gain": [weight_gain_value],
        "material_possessions": [material_possessions_value],
        "introvert": [introvert_value],
        "popping_up_stressful_memory": [popping_up_stressful_memory_value],
        "having_nightmares": [having_nightmares_value],
        "avoids_people_or_activities": [avoids_people_or_activities_value],
        "feeling_negative": [feeling_negative_value],
        "trouble_concentrating": [trouble_concentrating_value],
        "blamming_yourself": [blamming_yourself_value]
    }

    df_pre = pd.DataFrame(data)
    predictions = model.predict(df_pre)

    if predictions == 1:
        return {'message': 1}
    elif predictions == 0:
        return {'message': 0}
    elif predictions == 2:
        return {'message': 2}
    elif predictions == 3:
        return {'message': 3}
    elif predictions == 4:
        return {'message': 4}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
