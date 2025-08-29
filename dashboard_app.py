import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# --------------------------------------------------
# 設定
DATA_PATH = "data/sample_sales.csv"

# --------------------------------------------------
# データ読み込み
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 型変換
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    df["category"] = df["category"].str.strip()
    df["region"] = df["region"].str.strip()
    df["sales_channel"] = df["sales_channel"].str.strip()
    df["customer_segment"] = df["customer_segment"].str.strip()
    # 整合性確認用列
    df["revenue_calc"] = df["units"] * df["unit_price"]
    return df

df = load_data(DATA_PATH)

# --------------------------------------------------
# タイトル
st.title("販売データBIダッシュボード")

# --------------------------------------------------
# 日付範囲フィルタ
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "日付範囲を選択",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# フィルタ適用
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
else:
    df_filtered = df.copy()

# --------------------------------------------------
# KPI表示
total_revenue = df_filtered["revenue"].sum()
total_units = df_filtered["units"].sum()
total_categories = df_filtered["category"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("売上合計", f"¥{total_revenue:,.0f}")
col2.metric("販売数量合計", f"{total_units:,}")
col3.metric("商品カテゴリ数", f"{total_categories:,}")

# --------------------------------------------------
# カテゴリ別売上棒グラフ
category_sales = df_filtered.groupby("category", as_index=False)["revenue"].sum()
fig_bar = px.bar(
    category_sales,
    x="category",
    y="revenue",
    title="カテゴリ別売上",
    text="revenue"
)
fig_bar.update_traces(texttemplate="¥%{text:,.0f}", textposition="outside")
fig_bar.update_layout(yaxis_title="売上金額 (JPY)", xaxis_title="カテゴリ")
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# 日毎売上折れ線グラフ
daily_sales = df_filtered.groupby("date", as_index=False)["revenue"].sum()
fig_line = px.line(
    daily_sales,
    x="date",
    y="revenue",
    title="日別売上推移",
    markers=True
)
fig_line.update_layout(yaxis_title="売上金額 (JPY)", xaxis_title="日付")
st.plotly_chart(fig_line, use_container_width=True)
