import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# CSVファイルを読み込む（parse_datesは不要）
df = pd.read_csv('data/sample_sales.csv')

st.subheader('カテゴリ別合計売上グラフ')

# カテゴリ別の合計売上を集計
category,_revenue = df.groupby('category', as_index=False)['revenue'].sum()

# 棒グラフを作成
fig = px.bar(
    category_revenue,
    x='category',
    y='revenue',
    title='商品カテゴリごとの総売上',
    labels={'category': '商品カテゴリ', 'revenue': '総売上 (円)'}
)

st.plotly_chart(fig)

st.write('---')
st.write('このグラフはインタラクティブです！特定のカテゴリにカーソルを合わせると、そのカテゴリの正確な総売上が表示されます。')
