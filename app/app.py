import uvicorn
from fastapi import FastAPI
from loan_application import LoanApplication
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

model = pickle.load(open('../model/model_saved/bagging_v_0.1.pkl', 'rb'))


@app.get('/')
def index_page():
    return {'message': 'System OK'}


@app.post('/predict')
def loan_decision(data: LoanApplication):
    # retrieving data
    data = data.dict()
    loan_id = data['Loan_ID']
    gender = data['Gender']
    married = data['Married']
    dependents = data['Dependents']
    education = data['Education']
    self_employed = data['Self_Employed']
    applicant_income = data['Applicant_Income']
    coapplicant_income = data['CoapplicantIncome']
    loan_amount = data['Loan_Amount']
    loan_amount_term = data['Loan_Amount_Term']
    credit_history = data['Credit_History']
    property_area = data['Property_Area']

    # creating dataframe

    # data preparation for the model

    # predict data


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
