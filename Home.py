import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Streamlit BI x Claude Code Starter", layout="wide")

st.title("Streamlit BI x Claude Code Starter")
@st.cache_data
def load_data():
    orders_df = pd.read_csv("sample.data/orders.csv")
    users_df = pd.read_csv("sample.data/users .csv")
    return orders_df, users_df

@st.cache_data
def preprocess_data(orders_df):
    orders_df['created_at'] = pd.to_datetime(orders_df['created_at'])
    orders_df['year_month'] = orders_df['created_at'].dt.to_period('M')
    return orders_df

orders_df, users_df = load_data()
orders_df = preprocess_data(orders_df)

col1, col2 = st.columns(2)

with col1:
    st.metric("総注文数", f"{len(orders_df):,}")
    st.metric("総ユーザー数", f"{len(users_df):,}")

with col2:
    cancel_rate = len(orders_df[orders_df['status'] == 'Cancelled']) / len(orders_df) * 100
    st.metric("キャンセル率", f"{cancel_rate:.1f}%")
    avg_items = orders_df['num_of_item'].mean()
    st.metric("平均アイテム数", f"{avg_items:.1f}")

monthly_orders = orders_df.groupby('year_month').size().reset_index(name='order_count')
monthly_orders['year_month_str'] = monthly_orders['year_month'].astype(str)

fig_orders = px.bar(monthly_orders, x='year_month_str', y='order_count',
                   title='月別注文数',
                   labels={'year_month_str': '年月', 'order_count': '注文数'})
st.plotly_chart(fig_orders, use_container_width=True)

monthly_cancel = orders_df.groupby(['year_month', 'status']).size().unstack(fill_value=0)
monthly_cancel['cancel_rate'] = monthly_cancel.get('Cancelled', 0) / monthly_cancel.sum(axis=1) * 100
monthly_cancel = monthly_cancel.reset_index()
monthly_cancel['year_month_str'] = monthly_cancel['year_month'].astype(str)

fig_cancel = px.line(monthly_cancel, x='year_month_str', y='cancel_rate',
                    title='月別キャンセル率',
                    labels={'year_month_str': '年月', 'cancel_rate': 'キャンセル率 (%)'})
st.plotly_chart(fig_cancel, use_container_width=True)

st.header("注文データ（フィルタリング可能）")
status_filter = st.multiselect("ステータスでフィルタ", 
                              options=orders_df['status'].unique(),
                              default=orders_df['status'].unique())

filtered_orders = orders_df[orders_df['status'].isin(status_filter)]
st.dataframe(filtered_orders, use_container_width=True)

st.header("ユーザーデータ（上位10件）")
st.dataframe(users_df.head(10))