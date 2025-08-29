import os
import json
import re
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

st.set_page_config(page_title="売上データ分析AI（DuckDB×SQL）", layout="wide")
st.title("売上データ分析AI（DuckDB×SQL）")

# ===== OpenAI Client（タイムアウト/リトライ明示） =====
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # 必要に応じて変更
client = OpenAI(timeout=30, max_retries=2)

# ===== 接続ヘルスチェック =====
@st.cache_resource
def openai_healthcheck() -> tuple[bool, str]:
    try:
        # 軽い呼び出しで疎通確認（models.list）
        client.models.list()
        # 環境変数の存在も明示チェック
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY が設定されていません。"
        return True, ""
    except Exception as e:
        return False, f"OpenAIへの疎通に失敗しました（{type(e).__name__}）。ネットワーク/プロキシ/ファイアウォール/鍵をご確認ください。{e}"

ok, hc_msg = openai_healthcheck()
if not ok:
    st.error(hc_msg)

DATA_PATH = "data/sample_sales.csv"
TABLE_NAME = "sales"

@st.cache_resource
def get_conn():
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT
            CAST(date AS DATE) AS date,
            category AS category,
            CAST(units AS INTEGER) AS units,
            CAST(unit_price AS DOUBLE) AS unit_price,
            region AS region,
            sales_channel AS sales_channel,
            customer_segment AS customer_segment,
            COALESCE(CAST(revenue AS DOUBLE),
                     CAST(units AS DOUBLE) * CAST(unit_price AS DOUBLE)) AS revenue
        FROM read_csv_auto('{DATA_PATH}', header=True,
            columns={{'date':'DATE','category':'VARCHAR','units':'INTEGER','unit_price':'DOUBLE',
                     'region':'VARCHAR','sales_channel':'VARCHAR','customer_segment':'VARCHAR','revenue':'DOUBLE'}}
        );
    """)
    return con

con = get_conn()

# SQL validation patterns
DANGEROUS = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|ATTACH|COPY|PRAGMA)\b", re.I)

def validate_sql(sql: str) -> tuple[bool, str]:
    if not re.match(r"^\s*(WITH|SELECT)\b", sql or "", flags=re.I):
        return False, "最初の文はSELECTまたはWITHで始めてください。"
    if DANGEROUS.search(sql or ""):
        return False, "DDL/DMLは許可されていません。"
    if sql.count(";") > 1:
        return False, "複数ステートメントは不可です。"
    return True, ""

def run_sql(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

def auto_chart(df: pd.DataFrame, suggestion: dict):
    if df.empty:
        st.info("結果が空でした。条件を見直してください。")
        return
    
    cols = {c.lower(): c for c in df.columns}
    x = suggestion.get("x") or (cols.get("month") or cols.get("date") or df.columns[0])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    y = suggestion.get("y") or (num_cols[0] if num_cols else None)
    color = suggestion.get("color") or (cols.get("category") or cols.get("region") or cols.get("sales_channel"))
    
    chart_type = suggestion.get("type", "bar")
    
    try:
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color)
        elif chart_type == "area":
            fig = px.area(df, x=x, y=y, color=color)
        elif chart_type == "pie" and y is not None:
            fig = px.pie(df, names=color or x, values=y)
        else:
            fig = px.bar(df, x=x, y=y, color=color)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"グラフの描画でエラーが発生しました: {e}")
        # Fallback to simple table display
        st.dataframe(df, use_container_width=True)

def fallback_sql_and_show():
    sql = f"""
    SELECT category, SUM(revenue) AS revenue
    FROM {TABLE_NAME}
    GROUP BY category
    ORDER BY revenue DESC
    """
    st.code(sql, language="sql")
    df = run_sql(sql)
    st.success("接続エラーのためフォールバック集計を表示します（カテゴリ別売上）。")
    st.dataframe(df, use_container_width=True)
    auto_chart(df, {"type": "bar", "x": "category", "y": "revenue", "color": ""})

def generate_sql_via_llm(prompt: str) -> dict:
    SYSTEM_PROMPT = f"""
あなたはビジネスデータ分析用のSQL生成アシスタントです。
テーブルは {TABLE_NAME} のみ。DuckDB方言でSQLを出力。
出力はJSON一個のみ: {{"sql":"...","explanation":"...","chart_suggestion":{{"type":"auto|bar|line|area|pie","x":"","y":"","color":""}}}}
制約: SELECT/WITH開始、DDL/DML禁止、複数文禁止、月集計は date_trunc('month', date) AS month。
使用可能列: date, category, units, unit_price, region, sales_channel, customer_segment, revenue

例:
- 月毎×カテゴリ: SELECT date_trunc('month', date) AS month, category, SUM(revenue) AS total_revenue FROM sales GROUP BY month, category ORDER BY month, total_revenue DESC
- チャネル別売上: SELECT sales_channel, SUM(revenue) AS total_revenue FROM sales GROUP BY sales_channel ORDER BY total_revenue DESC
- 地域別売上合計: SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"ユーザーの質問: {prompt}\nJSONだけを返してください。"}
    ]
    
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
        stream=False,
    )
    return json.loads(resp.choices[0].message.content)

# Sidebar with data overview
with st.sidebar:
    st.header("📋 データ概要")
    try:
        # Show table info
        info_sql = f"SELECT COUNT(*) as row_count FROM {TABLE_NAME}"
        row_count = con.execute(info_sql).df().iloc[0]['row_count']
        st.success(f"✅ データ読み込み完了: {row_count}行")
        
        # Show sample data
        sample_sql = f"SELECT * FROM {TABLE_NAME} LIMIT 3"
        sample_df = con.execute(sample_sql).df()
        st.write("**サンプルデータ:**")
        st.dataframe(sample_df, use_container_width=True)
        
        # Show columns
        st.write("**カラム情報:**")
        st.write("- date (日付)")
        st.write("- category (カテゴリ)")
        st.write("- units (数量)")
        st.write("- unit_price (単価)")
        st.write("- region (地域)")
        st.write("- sales_channel (販売チャネル)")
        st.write("- customer_segment (顧客セグメント)")
        st.write("- revenue (売上)")
        
    except Exception as e:
        st.error(f"データベースエラー: {e}")

    # Quick query buttons
    st.header("🚀 クイック質問")
    if st.button("月毎×カテゴリ", use_container_width=True):
        st.session_state.quick_query = "月毎×カテゴリ別の売上を見せて"
    if st.button("チャネル別売上", use_container_width=True):
        st.session_state.quick_query = "チャネル別売上を教えて"
    if st.button("地域別売上合計", use_container_width=True):
        st.session_state.quick_query = "地域別売上合計を表示して"

# Main chat interface
st.header("💬 チャットインターフェース")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            st.code(message["sql"], language="sql")
        if message.get("dataframe") is not None:
            st.dataframe(message["dataframe"], use_container_width=True)

# Handle quick query
if hasattr(st.session_state, 'quick_query'):
    user_q = st.session_state.quick_query
    delattr(st.session_state, 'quick_query')
else:
    user_q = st.chat_input("売上データについて質問してください（例：月毎×カテゴリ、チャネル別、地域別 など）")

if user_q:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Generate assistant response
    with st.chat_message("assistant"):
        if not ok:
            st.warning("LLMに接続できないため、フォールバックを表示します。")
            fallback_sql_and_show()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "LLM接続エラーのため、フォールバック結果を表示しました。"
            })
        else:
            try:
                # Generate SQL using LLM
                with st.spinner('SQL生成中...'):
                    plan = generate_sql_via_llm(user_q)
                
                sql = (plan.get("sql") or "").strip()
                explanation = plan.get("explanation", "SQL実行結果です。")
                
                # Validate SQL
                valid, msg = validate_sql(sql)
                if not valid or not sql:
                    st.warning(f"SQL検証エラー: {msg}。フォールバックに切り替えます。")
                    fallback_sql_and_show()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"SQL検証エラー: {msg}。フォールバック結果を表示しました。"
                    })
                else:
                    # Display generated SQL
                    st.code(sql, language="sql")
                    
                    # Execute SQL
                    with st.spinner('SQL実行中...'):
                        df = run_sql(sql)
                    
                    if df.empty:
                        st.info("結果が空でした。条件を見直してください。")
                    else:
                        st.success(explanation)
                        st.dataframe(df, use_container_width=True)
                        
                        # Generate chart
                        with st.spinner('グラフ生成中...'):
                            chart_suggestion = plan.get("chart_suggestion", {"type": "auto", "x": "", "y": "", "color": ""})
                            auto_chart(df, chart_suggestion)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "sql": sql,
                        "dataframe": df if not df.empty else None
                    })
                    
            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                st.error(f"OpenAI接続/応答でエラー: {type(e).__name__} — {e}")
                fallback_sql_and_show()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"OpenAI接続エラー: {type(e).__name__}。フォールバック結果を表示しました。"
                })
            except json.JSONDecodeError as e:
                st.error(f"LLM応答のJSON解析エラー: {e}")
                fallback_sql_and_show()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "LLM応答の解析エラー。フォールバック結果を表示しました。"
                })
            except Exception as e:
                st.error(f"想定外のエラー: {e}")
                fallback_sql_and_show()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"想定外のエラー: {e}。フォールバック結果を表示しました。"
                })

# Clear chat history button
if st.button("チャット履歴をクリア", key="clear_history"):
    st.session_state.messages = []
    st.rerun()