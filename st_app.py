import streamlit as st
from datetime import datetime as dt
from src.stock_utilities import Pipeline


st.title("Stock Forecast")

st.header("Inputs")
tick = st.text_input('Ticker name (From Yahoo Finance)', 'MAXHEALTH.NS')

msg = "Start date, used for tuning the model parameters and back testing"
start_dt = st.date_input(msg, dt(year=2021, month=4, day=1))

msg = (
    "End date, model is tuned using data from start date to end date. "
    "Forecast will be done for the day after"
)
end_dt = st.date_input(msg, dt.today())

buy_price = st.number_input("At what price did you buy the above stock?", 0)

if st.button("Get recommendation!"):
    pipe = Pipeline(tick, start_dt, end_dt)
    if buy_price>0:
        df_pred, _, _, msg_dict = pipe.refresh_recommendation(buying_price=buy_price)
    else:
        df_pred, _, _, msg_dict = pipe.refresh_recommendation()

    st.header("Results")
    st.dataframe(df_pred)
    st.subheader("Recommendation")
    st.text(msg_dict['recom'])
    st.subheader("Strategy 1 backtesting result")
    st.text(msg_dict['additional_info'])