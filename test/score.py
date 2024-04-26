#!/usr/bin/env python
# coding: utf-8

from azureml.core.model import Model
import joblib
import pandas as pd



def init():
    model_path = Model.get_model_path(model_name="price_car_data.pkl")
    model = joblib.load(model_path)
    return model

def predict(model):
    new_car = pd.DataFrame({'name': [2], 'year': [2015], 'km_driven': [80000]})
    predicted_price = model.predict(new_car)
    print("Predicted selling price for the 7car: ", predicted_price)

def main():

    model = init()
    predict(model)

if __name__ == '__main__':
    main()