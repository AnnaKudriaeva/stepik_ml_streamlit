import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict

df = pd.read_csv("data/cleaned_carsData.csv")

def process_main_page():
    show_main_page()
    process_side_bar_inputs()

def show_main_page():
    image = Image.open('data/car.jpeg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_icon=image,

    )

    st.write(
        """
        # Car Price Prediction
        ## Are you planning to sell your car !?
        So let's try evaluating the price..
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_features_df, _ = split_data(train_df)
    full_features_df = pd.concat((user_input_df, train_features_df), axis=0)
    

    preprocessed_features_df = preprocess_data(full_features_df, test=False)

    user_features_df = preprocessed_features_df[:1]
    write_user_data(user_features_df)

    prediction, prediction_probas = load_model_and_predict(user_features_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    p1 = st.sidebar.selectbox("Бренд", df['brand'].unique())

    p4 = st.sidebar.slider("Год производства", min_value=1990, max_value=2020,
                            step=1)
    
    p5 = st.sidebar.slider("Пробег на дату продажи", min_value=0, 
                            max_value=1000000, step=25000)

    p6 = st.sidebar.selectbox("Количество мест",
                                df['seats'].unique())
    
    p7 = st.sidebar.selectbox("Количество владельцев",
                                df['owner'].unique())
    
    p8 = st.sidebar.selectbox("Короюка передач",
                                df['transmission'].unique())
    
    p9 = st.sidebar.selectbox("Продавец",
                                df['seller_type'].unique())
    
    p10 = st.sidebar.selectbox("Тип топлива",
                                options = df['fuel'].unique())
    
    p11 = st.sidebar.slider("Пробег",
                                min_value=0, max_value=55, step=5)
    
    p12 = st.sidebar.slider("Рабочий объем двигателя",
                                min_value=600, max_value=3000, step=200)
    
    p13 = st.sidebar.slider("Пиковая мощность двигателя",
                                min_value=30, max_value=300, step=30)
    
    p14 = st.sidebar.slider("Крутящий момент",
                                min_value=30, max_value=1500, step=50)
    
    p15 = st.sidebar.slider("Крутящий момент, максимальный",
                                min_value=500, max_value=5000, step=500)

    data = {
        "brand": p1,
        "year": p4,
        "km_driven": p5,
        "seats": p6,
        "owner": p7,
        "transmission": p8,
        "seller_type": p9,
        "fuel": p10,
        'mileage': p11,
        'engine': p12,
        'max_power': p13,
        'torque_nm': p14,
        'torque_max_rpm': p15
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()