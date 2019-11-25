import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
f=open('data7_name.csv','r')
attributes= f.readline().split(',')
heartDisease=pd.read_csv('data7.csv',names=attributes)
print("\nAttributes and datatypes")
print(heartDisease.dtypes)
model=BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),('fbs','heartdisease')])
model.fit(heartDisease,BayesianEstimator)
HeartDisease_infer=VariableElimination(model)
print("\n 1. Probability heart disease given age=28")
q=HeartDisease_infer.query(['heartdisease'],{'age':28})
print(q['heartdisease'])
print("\n 2. Probability of heart disease for male")
q=HeartDisease_infer.query(['heartdisease'],{'sex':1})
print(q['heartdisease'])