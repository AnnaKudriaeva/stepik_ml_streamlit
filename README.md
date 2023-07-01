# stepik_ml_streamlit

# Определение стоимости автомобилей

## Описание
Задача регрессии. В проекте рассмотрены модели градиентного бустинга в сравнении с более простыми - по времени обучения, скорости и качеству предсказания. Также особое внимание уделено обработке пропусков и выбросов. Представить результат в виде работающего веб-приложения с помощью фреймворка Streamlit.

## Данные
В наличии были следующие данные об автомобилях:
#### Целевая переменная
*selling_price: цена продажи, числовая

#### Признаки
*name (string): модель автомобиля  
*year (numeric, int): год выпуска с завода-изготовителя  
*km_driven (numeric, int): пробег на дату продажи  
*fuel (categorical: Diesel или Petrol, или CNG, или LPG, или electric): тип топлива  
*seller_type (categorical: Individual или Dealer, или Trustmark Dealer): продавец  
*transmission (categorical: Manual или Automatic): тип трансмиссии  
*owner (categorical: First Owner или Second Owner, или Third Owner, или Fourth & Above Owner): какой по счёту хозяин?  
*mileage (string, по смыслу числовой): пробег, требует предобработки  
*engine (string, по смыслу числовой): рабочий объем двигателя, требует предобработки  
*max_power (string, по смыслу числовой): пиковая мощность двигателя, требует предобработки  
*torque (string, по смыслу числовой, а то и 2): крутящий момент, требует предобработки  
*seats (numeric, float; по смыслу categorical, int)  
 
## Задача
Построить модель для быстрого определения стоимости автомобиля с пробегом по техническим характеристикам и комплектации. Представить результат в виде работающего веб-приложения с помощью фреймворка Streamlit.

## Статус
Завершен

## Результат
Самый лучший результат при кросс-валидации (и в итоге на тестовой выборке) дает метод CatBoostRegressor, имеет самую высокую точность предсказания, однако, дольше всего обучается.

## Используемые библиотеки
*Pandas, Scikit-learn, Catboost, LightGBM*

## Ссылка на веб-приложение:
https://carstepikmlapp.streamlit.app/
