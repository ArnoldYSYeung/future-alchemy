# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:45:52 2020

@author: Arnold
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
from datetime import datetime
import json


from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


n_future_days = 7
label_name = "Future"


def fit_polynomial(df, degree, n_days, target_date):
    
    poly_df = df.copy()
    
    last_date_idx = df['Date'].last_valid_index()
    last_date = df['Date'].iloc[last_date_idx]
    last_date = datetime.strptime(last_date, '%Y-%m-%d')
    
    target_date_s = datetime.strftime(target_date, '%Y-%m-%d')
    last_day = poly_df.tail(1)['day'].item()

    weekends = [5,6]
    while last_date < target_date + dt.timedelta(days=n_future_days):
        last_date = last_date + dt.timedelta(days=1)
        if last_date.weekday() not in weekends:
            last_date_idx += 1
            last_day += 1
            poly_df = pd.concat([poly_df, pd.DataFrame([[np.nan] * df.shape[1]], columns=poly_df.columns)], ignore_index=True)
            poly_df['Date'].iloc[last_date_idx] = last_date.strftime("%Y-%m-%d")
            poly_df['day'].iloc[last_date_idx] = last_day
    
    if predict_date.strftime("%Y-%m-%d") not in poly_df['Date'].unique():
        st.write("This is a weekend or something.  Missing data.")
        
        
    target_day = poly_df[poly_df['Date'] == target_date.strftime("%Y-%m-%d")]['day'].item()
        
    
    end_day = target_day - n_future_days
    start_day = end_day - n_days - 2 
    
    dates = poly_df[poly_df['day'] <= target_day]
    dates = dates[dates['day'] >= start_day]['Date']
    poly_df = poly_df[poly_df['day'] <= end_day]
    poly_df = poly_df[poly_df['day'] >= start_day]
        
    x = poly_df['day']
    y = poly_df['Close']
    
    poly_model = np.poly1d(np.polyfit(x, y, degree))
    
    #day_range = np.linspace(np.min(x), np.max(x), np.max(x)-np.min(x)+1)
    
    plt.scatter(x, y)
    interpolation = poly_model(x)
    plt.plot(x, interpolation) 
    
    #st.write(y)
    
    mae = mean_absolute_error(y, interpolation)
    st.write("Fitting MAE: ", mae)
    
    new_day_range = np.linspace(np.max(x)+1, np.max(x)+n_future_days, n_future_days)
    extrapolation = poly_model(new_day_range)
    plt.plot(new_day_range, extrapolation)
    
    predicted_values = np.concatenate((interpolation, extrapolation), axis=0)

    dates = dates.reset_index().drop(columns=["index"])
    
    st.write(dates.join(pd.DataFrame({'Prediction': predicted_values}).reset_index())
             .drop(columns=['index']))
    
    #plt.xticks(ticks=list(range(int(np.min(x)), int(np.min(x))+len(dates.to_list()))), 
    #           labels=dates.tolist(), rotation=90)
    
    #ticks = x['day'].to_list() + new_day_range
    new_day_range = pd.Series(new_day_range)
    ticks = pd.concat([x, new_day_range], axis=0)
    plt.xticks(ticks=ticks, labels=dates['Date'].to_list(), rotation=90)
    
    st.pyplot(bbox_inches="tight")

    st.write("The predicted price for ", target_date, " is ", 
                 poly_model(target_day))

def load_model(model_type):
    
    scaler = joblib.load("./models/scaler.pkl")
    
    model_file = "./models/"+str(model_type)+".pkl"
    
    try:    
        model = joblib.load(model_file)
    except:
        st.write("Cannot load file.  Please make sure Scikit-Learn is 0.22.2.")
        raise ValueError("File incompatible with Scikit-Learn version.")
    st.write(type(model))
    return model, scaler

def predict_future_price(model, scaler, df, predict_date):
    
    #predict_df = df[df['Date'] == predict_date]
    predict_df = df.copy()
        
    date_series = predict_df['Date']
    
    predict_df = predict_df.drop(columns=['Date', label_name])
    predict_np = scaler.transform(predict_df)
    prediction = model.predict(predict_np)
    
    prediction_df = pd.DataFrame({'Prediction': prediction})
    prediction_df.index = prediction_df.index+n_future_days
    predict_df = pd.concat([date_series, predict_df, prediction_df], axis=1)
        
    #   add missing dates
    weekends = [5,6]
    extra_dates = []
    
    start_date_idx = predict_df['Date'].last_valid_index()
    start_date = predict_df['Date'].iloc[start_date_idx]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    while len(extra_dates) < n_future_days:
        start_date = start_date + dt.timedelta(days=1)
        if start_date.weekday() not in weekends:
            start_date_idx += 1
            predict_df['Date'].iloc[start_date_idx] = start_date.strftime("%Y-%m-%d")
            extra_dates.append(start_date.strftime("%Y-%m-%d"))
    
    if predict_date.strftime("%Y-%m-%d") in predict_df['Date'].unique():
        st.write("The predicted price for ", predict_date, " is ", 
                 predict_df[predict_df['Date'] == predict_date.strftime("%Y-%m-%d")]['Prediction'])
        original_idx = predict_df.index[predict_df['Date'] == predict_date.strftime("%Y-%m-%d")] - n_future_days
        
        st.write("This value was predicted from ", predict_df.iloc[original_idx]['Date'])
        
    else:
        st.write("LOL...can't find this date. Probably a weekend or too far into the future (or past).")
    return predict_df

def show_explanation(model_type, model, scaler, feat_names, df, predict_date):
    
    if model_type in ["ridge", "lasso"]:
        
        coefficients = pd.DataFrame({"Coeff": model.coef_})
        feat_df = pd.DataFrame({"Features": feat_names})
        importance_df = pd.concat([feat_df, coefficients], axis=1)
        
        fig = plt.figure()

        #   get intercept
        st.write("The intercept (base price) is ", model.intercept_)

        #   compute the addition/subtraction of each feature
        
        #   get row corresponding to predict_date features
        original_idx = df.index[df['Date'] == predict_date.strftime("%Y-%m-%d")] - n_future_days
        feats = df.iloc[original_idx].drop(columns=['Date', "Prediction"])
        feats = scaler.transform(feats)
        
        addition_df = importance_df.join(pd.DataFrame({"score": np.transpose(feats)[:,0]}))
        addition_df['add_to_price'] = addition_df['Coeff'] * addition_df['score']
        st.write(addition_df.sort_values(by="add_to_price", ascending=False))
        st.write("The summed change from features is ", addition_df['add_to_price'].sum())

        importance_df = importance_df.sort_values(by="Coeff")            
        orig_importance_df = importance_df

        #   remove coeff = 0
        removed_feats = importance_df[importance_df['Coeff'] == 0]['Features'].to_list()
        importance_df = importance_df[importance_df['Coeff'] != 0]
        
        num_rows = importance_df.shape[0]
        if importance_df.shape[0] > 20:
            importance_df = pd.concat([importance_df.head(10), importance_df.tail(10)], axis=0) 
            st.write('Top 20 Feature Weights (relative importance)')
        else:
            st.write('Feature Weights (relative importance): ')
        importance_df.plot.barh(x='Features', y="Coeff")
        explain_plot = st.pyplot(bbox_inches="tight")   
        
        if model_type is "lasso":
            st.write("The following features have no effect: ", removed_feats)
        
        #   output matrix
                    
        #plt.xticks(y_pos, importance_df['Features'])
        #plt.xticks(rotation=45)
        
    

def plot_date_linegraph(df, series=[], sec_series=[]):
    
    fig = plt.figure()
    ax = df.reset_index().plot(x='Date', y=series)
    if sec_series != [] and sec_series != [""]:
        ax2 = df.plot(x='Date',y=sec_series,secondary_y=True, ax=ax)

    plt.xticks(rotation=45)
    ax.xaxis.set_tick_params(rotation=45)
    if sec_series != [] and sec_series != [""]:
        ax2.xaxis.set_tick_params(rotation=45)
    plt.xlabel("Date")
    
    main_plot = st.pyplot(bbox_inches = "tight")
    df.set_index('Date')

if __name__ == "__main__":
    st.title("ALCHEMY - Gold Price 7-Day Forecast")

    st.write("Using the USD and Bitcoin to predict gold price trajectory")
    
    df = pd.read_csv("feats_joined_data_df.csv")
    train_df = pd.read_csv("train_df.csv")
    valid_test_df = pd.read_csv("valid_test_df.csv")
    feat_names = train_df.columns.to_list()
    
    test_scores = json.load(open("top_model_configs.json"))
    test_scores = pd.DataFrame(test_scores).dropna(axis=0, how='any')
    
    start_train_date = train_df['Date'].iloc[train_df['Date'].first_valid_index()]
    end_train_date = train_df['Date'].iloc[train_df['Date'].last_valid_index()]
    st.write("Train Period: ", start_train_date, " to ", end_train_date)
    start_test_date = valid_test_df['Date'].iloc[valid_test_df['Date'].first_valid_index()]
    end_test_date = valid_test_df['Date'].iloc[valid_test_df['Date'].last_valid_index()]
    st.write("Valid-Test Period: ", start_test_date, " to ", end_test_date)
    if st.checkbox("Show estimated mean absolute errors for ML models (price)"):
        st.write(test_scores)
    
    if st.checkbox("List all available features"):
        st.write(feat_names)
    
    predict_df = None

    cols_str = st.text_input("Primary features to plot: ", "Close, boll_hband, boll_lband")
    cols_str2 = st.text_input("Secondary features to plot: ")

    if st.button("Plot"):
        cols = cols_str.replace(" ", "").split(",")
        cols2 = cols_str2.replace(" ", "").split(",")
        #plot_date_linegraph(df, ['Close', 'boll_hband', 'boll_lband'])
        plot_date_linegraph(df, cols, cols2)
    
    if st.checkbox('Show Existing DataFrame'):
        st.write(df)
    
    st.write("Predict the 7-day price.")
    
    degree = st.sidebar.slider("Polynomial order", 1, 10, 3)
    poly_daterange = st.sidebar.slider("Polynomial date range", n_future_days, df.shape[0], 30)
    
    model_type = st.sidebar.selectbox('Select a ML model', ('lasso',
                                                           'ridge', 
                                                           'SVR',
                                                           'rf',
                                                           'nn'))
    
    explain = st.sidebar.checkbox("Show Explanation")
    
    date_str = st.text_input("Enter date to predict (e.g., 2020-07-15):")
    
    if st.button("Run Polynomial Fitting"):
        if date_str:
            predict_date = datetime.strptime(date_str, '%Y-%m-%d')
            fit_polynomial(df, degree, poly_daterange, predict_date)
    
    
    
    if st.button("Run ML Model"):
        st.write("Loading model...")
        model, scaler = load_model(model_type)
        st.write("Model loaded.")
        st.write("Train Period: ", start_train_date, " to ", end_train_date)
        st.write("Valid-Test Period: ", start_test_date, " to ", end_test_date)
        if date_str:
            predict_date = datetime.strptime(date_str, '%Y-%m-%d')
            predict_df = predict_future_price(model, scaler, df, predict_date)
            
            cols = cols_str.replace(" ", "").split(",")
            cols2 = cols_str2.replace(" ", "").split(",")
            
            plot_date_linegraph(predict_df, cols+['Prediction'], cols2)
            st.write("Prediction: ", predict_df)
            
            #   show explanation
            if explain:
                feat_names.remove("Date")
                feat_names.remove(label_name)
                show_explanation(model_type, model, scaler, feat_names, 
                                 predict_df, predict_date)


                    
    
    