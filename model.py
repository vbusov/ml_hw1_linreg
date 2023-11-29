import pickle

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


OHE = "pickles/preprocessor.pickle"
MODELS = {
    "ridge": "pickles/ridge.pickle",
}
USE_MODEL = "ridge"


def load_object(filename: str): 
    with open(filename, "rb") as f:
        return pickle.load(f)


def preprocess(input_objects):
    df = pd.DataFrame(input_objects)
    for column in ["engine", "mileage", "max_power"]:
        num_regex = r'(\d*\.\d+|\d+)'
        df[column] = df[column].str.extract(num_regex, expand=False).astype(float)
    for column in ["seats", "engine"]:
        df[column] = df[column].astype(int)
    df.drop("torque", axis=1, inplace=True)
    
    ohe = load_object(OHE)
    columns = ohe.get_feature_names_out()
    return pd.DataFrame(ohe.transform(df), columns=columns)


def predict(data, use_model=USE_MODEL):
    predictor = load_object(MODELS[use_model])
    return predictor.predict(data)
