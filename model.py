import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostRegressor
from pickle import dump, load

RANDOM_STATE = 12345


def open_data(path="/data/car_prediction_train.csv"):
    df = pd.read_csv(path)
    df = df

    return df 

def split_data(df: pd.DataFrame):
    target = df['selling_price']
    features = df.drop(['selling_price'], axis=1)

    return features, target

def fit_and_save_model(features, target, path="/data/finalized_model.mw"):
    model = CatBoostRegressor()

    model.fit(features, target)
    predicted_test = model.predict(features)
    rmse = round(mean_squared_error(target, predicted_test)**0.5, 3)
    print(f"Model rmse is {rmse}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="/data/finalized_model.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вам не повезло с вероятностью",
        1: "Вы выживете с вероятностью"
    }

    encode_prediction = {
        0: "Сожалеем, вам не повезло",
        1: "Ура! Вы будете жить"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df

if __name__ == "__main__":
    df = open_data()
    features, target = split_data(df)
    fit_and_save_model(features, target)