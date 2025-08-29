import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# CSVファイルを読み込む
df = pd.read_csv('data/sample_sales.csv')

# ---- ここから変更：日毎の売上推移（折れ線・赤） ----
st.subheader('日毎の売上推移グラフ')

# 日付列をdatetimeに変換（列名が違う場合は "date" を修正）
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 日付がNaTの行を除外
df = df.dropna(subset=['date'])

# 日毎の合計売上を集計
daily_revenue = (
    df.groupby(df['date'].dt.date)['revenue']
      .sum()
      .reset_index()
      .rename(columns={'date': '日付', 'revenue': '売上'})
)

# 折れ線グラフを作成（線の色は赤）
fig = px.line(
    daily_revenue,
    x='日付',
    y='売上',
    title='日毎の売上推移',
    labels={'日付': '日付', '売上': '売上 (円)'},
    color_discrete_sequence=['red']
)

# 太さなど微調整（任意）
fig.update_traces(line=dict(width=3))

# x軸を日付として見やすく
fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"))

# 表示
st.plotly_chart(fig, use_container_width=True)
# ---- 変更ここまで ----

st.write('---')
st.write('このグラフはインタラクティブです！ポイントにカーソルを合わせると、その日の正確な売上が表示されます。')
