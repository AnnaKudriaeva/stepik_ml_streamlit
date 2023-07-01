import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from catboost import CatBoostRegressor
from pickle import dump, load

RANDOM_STATE = 12345

def split_data(df: pd.DataFrame):
    target = df['selling_price']
    features = df.drop(['selling_price'], axis=1)

    return features, target

def open_data(path="data/cleaned_carsData.csv"):
    df = pd.read_csv(path)
    df = df[['selling_price', 'brand', "year", "km_driven",
        "seats", "owner", "transmission", "seller_type",
        "fuel", 'mileage', 'engine', 'max_power',
        'torque_nm','torque_max_rpm']]

    return df

def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        features_df, target_df = split_data(df)
    else:
        features_df = df

    categorial_RF = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    numeric = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque_nm', 'torque_max_rpm', 'seats']
    scaler = StandardScaler()
    scaler.fit(df[numeric])
    features_df[numeric] = scaler.transform(df[numeric])
    pd.options.mode.chained_assignment = None

    encoder_ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    encoder_ohe.fit(features_df[categorial_RF])

    tmp = pd.DataFrame(encoder_ohe.transform(features_df[categorial_RF]).toarray(),
                            columns=encoder_ohe.get_feature_names_out(),
                            index=features_df.index)
    
    features_df.drop(categorial_RF, axis=1, inplace=True)
    features_df = features_df.join(tmp)

    if test:
        return features_df, target_df
    else:
        return features_df

def fit_and_save_model(features_df, target_df, path="data/finalized_model.mw"):
    model = CatBoostRegressor()

    model.fit(features_df, target_df)
    predicted_test = model.predict(features_df)
    rmse = round(mean_squared_error(target_df, predicted_test)**0.5, 3)
    accuracy = accuracy_score(predicted_test, target_df)
    print(f"Model accuracy is {accuracy}")
    print(f"Model rmse is {rmse}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="data/finalized_model.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)
    
    return prediction

if __name__ == "__main__":
    df = open_data()
    features, target = split_data(df)
    fit_and_save_model(features, target)