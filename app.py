import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from model import open_data, split_data, load_model_and_predict

carsData = pd.read_csv("data/cleaned_carsData.csv")

def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/car.jpeg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Car Price Prediction",
        page_icon=image,

    )

    st.write(
        """
        # Are you planning to sell your car !?
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
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    

    
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    p1 = st.sidebar.selectbox("Бренд", options = carsData['brand'].unique())
    
    p4 = st.sidebar.slider("Год производства", min_value=1990, max_value=2020,
                            step=1)

    p5 = st.sidebar.slider(
        "Пробег на дату продажи",
        min_value=0, max_value=1000000, step=25000)

    p6 = st.sidebar.selectbox("Количество мест",
                                options = carsData['seats'].unique())
    
    p7 = st.sidebar.selectbox("Количество владельцев",
                                options = carsData['owner'].unique())
    
    p8 = st.sidebar.selectbox("Короюка передач",
                                options = carsData['transmission'].unique())
    
    p9 = st.sidebar.selectbox("Продавец",
                                options = carsData['seller_type'].unique())
    
    p10 = st.sidebar.selectbox("Тип топлива",
                                options = carsData['fuel'].unique())
    
    p11 = st.sidebar.selectbox("Пробег",
                                options = carsData['mileage'].unique())
    
    p12 = st.sidebar.selectbox("Рабочий объем двигателя",
                                options = carsData['engine'].unique())
    
    p13 = st.sidebar.selectbox("Пиковая мощность двигателя",
                                options = carsData['max_power'].unique())
    
    p13 = st.sidebar.selectbox("Количество мест",
                                options = carsData['max_power'].unique())
    
    p14 = st.sidebar.selectbox("Крутящий момент",
                                options = carsData['torque_nm'].unique())
    
    p15 = st.sidebar.selectbox("Крутящий момент, максимальный",
                                options = carsData['torque_max_rpm'].unique())

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