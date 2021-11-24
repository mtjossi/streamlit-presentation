import streamlit as st
import pandas_datareader.data as pdr
import yfinance as yf
import plotly.express as px
import datetime as dt
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

st.header('Streamlit Quick Showcase - \n Hello ZW, FH, KL, XL & co')
st.sidebar.subheader('Please Select a Page')
page_selection = st.sidebar.radio('Choose a Page:', ['Page 1: Download Financial Data', 'Page 2: Show Financials', 'Page 3: Quick ML'])
page_dict = dict(zip(['Page 1: Download Financial Data', 'Page 2: Show Financials', 'Page 3: Quick ML'], ['P1', 'P2', 'P3']))
yf.pdr_override()

@st.cache
def convert_df_csv(df):
    csv_file = df.to_csv().encode('utf-8')
    return csv_file

if page_dict[page_selection] == 'P1':
    ticker = st.text_input("Please enter a ticker: (e.g. TSLA)")
    start = st.date_input("Please enter a starting date:", (dt.datetime.today() - dt.timedelta(30)), max_value=(dt.datetime.today() - dt.timedelta(1)))
    end = st.date_input("Please enter an end date:", dt.datetime.today(), max_value=dt.datetime.today())
    if start >= end:
        st.write("Please make sure end date is after start date...")
    else:
        if ticker:
            df = pdr.get_data_yahoo([t.strip() for t in ticker.split(',')], start, end)
            df.index = pd.to_datetime(df.index)
            st.write(f"{ticker} data from {start} to {end}")
            st.dataframe(df.head(30))

            fig = px.line(df, x=df.index, y='Adj Close', title=f"{ticker} - Adjusted Close Graph")
            st.plotly_chart(fig)

            data = convert_df_csv(df)
            st.download_button(label='📥 Download Data as csv',
                                data=data,
                                file_name= f'{ticker}_data.csv',
                                mime='text/csv',
                                key='download-csv')

if page_dict[page_selection] == 'P2':
    ticker = st.text_input("Please enter a ticker: (e.g. TSLA)")
    st.write(f"{ticker} info")

    info = yf.Ticker(ticker)
    st.write('Easily display dataframes, dictionaries, images etc.')
    st.image(info.info['logo_url'])
    st.write(f"{ticker} balance sheet")
    st.write(info.balance_sheet)
    st.write(f"{ticker} financials")
    st.write(info.financials)
    st.write(f"{ticker} info")
    st.write(info.info)

if page_dict[page_selection] == 'P3':

    dataset_choice = st.selectbox('Choose a dataset', ['Wisconsin Breast Cancer', 'Iris'])
    if dataset_choice == 'Wisconsin Breast Cancer':
        data, target = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
        st.write("Simple classification on Wisconsin Breast Cancer dataset")
        X_train, X_test, y_train, y_test = train_test_split(data, target)

        st.write("Features:")
        st.dataframe(data.sample(5))
        st.write("Target (0 for benign, 1 for malignant)")
        st.dataframe(target.sample(5))
    
    if dataset_choice == 'Iris':
        data, target = datasets.load_iris(return_X_y=True, as_frame=True)
        st.write("Simple classification on Iris dataset")
        X_train, X_test, y_train, y_test = train_test_split(data, target)

        st.write("Features:")
        st.dataframe(data.sample(5))
        st.write("Target (0 for setosa, 1 for versicolor, 2 for virginica)")
        st.dataframe(target.sample(5))

    if st.button("Run Classification Model:"):
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write("Confusion matrix")
        mat = confusion_matrix(y_test, y_pred)
        fig, ax = plot_confusion_matrix(mat, show_normed=True)
        st.pyplot(fig)


