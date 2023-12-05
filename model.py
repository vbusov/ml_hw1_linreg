import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


OHE = "pickles/preprocessor.pickle"
MODEL = "pickles/model.pickle"


def load_object(filename: str): 
    with open(filename, "rb") as f:
        return pickle.load(f)


def preprocess(input_objects):
    df = pd.DataFrame(input_objects)

    for column in ["engine", "mileage", "max_power"]:
        num_regex = r'(\d*\.\d+|\d+)'
        df[column] = df[column].str.extract(num_regex, expand=False).astype(float)
    df.drop(["torque"], axis=1, inplace=True)

    for column in ["seats", "engine"]:
        df[column] = df[column].astype(int)

    # extract maker and model info from car name
    df["maker"] = df["name"].str.split(expand=True)[0]
    df["model"] = df["name"].str.split(expand=True)[1]
    df["model"] = df["model"].str.lower()

    # construct car age and take log
    df["car_age"] = df["year"].max() - df["year"]
    df["car_age_log"] = np.log(df["car_age"] + 1)

    # take log of km_driven
    df["km_driven_log"] = np.log(df["km_driven"] + 1)
    df["mileage_log"] = np.log(df["mileage"] + 1)

    # combine features
    df["max_power_x_engine"] = df["max_power"] * df["engine"]
    df["max_power_x_engine_log"] = np.log(df["max_power_x_engine"] + 1)

    df["car_age_log_x_km_driven_log"] = df["car_age_log"] * df["km_driven_log"]
    df["car_age_log_x_km_driven_log_log"] = np.log(df["car_age_log_x_km_driven_log"] + 1)

    df["car_age_log_x_mileage_log"] = df["car_age_log"] * df["mileage_log"]
    df["car_age_log_x_mileage_log_log"] = np.log(df["car_age_log_x_mileage_log"] + 1)
    
    ohe = load_object(OHE)
    columns = ohe.get_feature_names_out()

    df = ohe.transform(df)
    if ohe.sparse_output_:
      df = df.toarray()
    df = pd.DataFrame(df, columns=columns)

    return df


def predict(data, use_model=MODEL):
    predictor = load_object(MODEL)
    return predictor.predict(data)
