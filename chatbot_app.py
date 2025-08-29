import streamlit as st
from openai import OpenAI
import pandas as pd
import os
# API キーの確認
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEYが設定されていません。環境変数を確認してください。")
    st.stop()

client = OpenAI(api_key=api_key, timeout=60.0, base_url="https://api.openai.com/v1")

@st.cache_data
def load_sales_data():
    data_path = os.path.join("data", "sample_sales.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

sales_data = load_sales_data()

# アプリのタイトルを設定します
st.title('シンプルなAIチャットボット')

if "messages" not in st.session_state:
    st.session_state.messages = []

if sales_data is not None:
    st.sidebar.markdown("### 📊 売上データ情報")
    st.sidebar.markdown(f"**データ期間:** {sales_data['date'].min()} ～ {sales_data['date'].max()}")
    st.sidebar.markdown(f"**総レコード数:** {len(sales_data):,} 件")
    st.sidebar.markdown(f"**カテゴリ数:** {sales_data['category'].nunique()} 種類")
    st.sidebar.markdown("**利用可能なデータ:**")
    st.sidebar.markdown("- 日付、カテゴリ、数量、単価")
    st.sidebar.markdown("- 地域、販売チャネル、顧客セグメント")
    st.sidebar.markdown("- 売上高")
else:
    st.sidebar.error("売上データが読み込めませんでした")

# これまでのチャット履歴を表示します
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの新しいメッセージを受け取る入力欄を表示します
if prompt := st.chat_input("何か質問してください..."):
    # ユーザーのメッセージを履歴に追加して表示します
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        system_message = {
            "role": "system", 
            "content": """あなたは売上データ分析の専門家です。以下の売上データを使って質問に答えてください。

利用可能なデータ:
- date: 日付 (YYYY-MM-DD形式)
- category: 商品カテゴリ (Electronics, Groceries, Clothing, Home & Kitchen, Sports, Beauty)
- units: 販売数量
- unit_price: 単価
- region: 地域 (North, South, East, West)
- sales_channel: 販売チャネル (Online, Store)  
- customer_segment: 顧客セグメント (Consumer, Corporate, Small Business)
- revenue: 売上高

データ分析や集計が必要な質問には具体的な数値で回答し、グラフや表を作成できる場合は提案してください。"""
        }

        if sales_data is not None:
            system_message["content"] += f"\n\n現在のデータ概要:\n- 期間: {sales_data['date'].min()} ～ {sales_data['date'].max()}\n- レコード数: {len(sales_data):,} 件\n- 総売上: ¥{sales_data['revenue'].sum():,}"

        messages = [system_message] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        try:
            # まずシンプルな非ストリーミングリクエストを試す
            response_obj = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=False,
            )
            response = response_obj.choices[0].message.content
            st.write(response)
        except Exception as e:
            st.error(f"APIエラーが発生しました: {str(e)}")
            st.error(f"詳細: {type(e).__name__}")
            response = f"エラー: {str(e)}"
    st.session_state.messages.append({"role": "assistant", "content": response})