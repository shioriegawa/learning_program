# -*- coding: utf-8 -*-
"""
Streamlit オリジナルBIダッシュボード（初期版）
- 参照データ: data/sample_sales.csv
- 日本語UI / 円表記 / 基本KPI・時系列・構成比・クロス集計・テーブル・ダウンロード

実装ポイント（抜粋）
- スキーマ検証（必須8列）
- 前処理: 日付変換, revenue補完, 重複集約, 時間列, 外れ値フラグ
- キャッシュ: 読み込み・前処理
- UI: サイドバー（フィルタ・設定・品質指標・DL）, メイン（KPI→グラフ→テーブル）
- グラフ: Altair（折れ線/棒/ドーナツ/ヒートマップ）
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------------------------------------------
# 基本設定
# -------------------------------------------------------------
st.set_page_config(
    page_title="売上ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Altair設定
alt.data_transformers.disable_max_rows()

# -------------------------------------------------------------
# 定数・期待スキーマ
# -------------------------------------------------------------
DATA_PATH = Path("data/sample_sales.csv")
REQUIRED_COLS = [
    "date",
    "category",
    "units",
    "unit_price",
    "region",
    "sales_channel",
    "customer_segment",
    "revenue",
]

# -------------------------------------------------------------
# ユーティリティ
# -------------------------------------------------------------

def yen(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"¥{int(round(x)):,}"
    except Exception:
        return "—"


def intfmt(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return "—"


def pctfmt(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{x:.{digits}%}"
    except Exception:
        return "—"


# -------------------------------------------------------------
# データ読み込み・前処理
# -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """CSVを読み込み。無ければサンプルでフォールバック。"""
    if path.exists():
        df = pd.read_csv(path)
    else:
        st.warning("参照ファイルが見つからないため、サンプルデータで表示します。")
        sample = io.StringIO(
            """
            date,category,units,unit_price,region,sales_channel,customer_segment,revenue
            2025-01-01,Electronics,7,19990,North,Online,Small Business,139930
            2025-01-01,Groceries,7,250,East,Online,Consumer,1750
            2025-01-01,Clothing,5,3990,South,Online,Small Business,19950
            2025-01-01,Home & Kitchen,6,5990,North,Store,Consumer,35940
            2025-01-02,Electronics,3,14990,West,Store,Consumer,44970
            2025-01-03,Sports,4,2990,North,Online,Consumer,11960
            """.strip()
        )
        df = pd.read_csv(sample)
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """型変換・補完・重複集約・時間列・外れ値フラグ等を付与。品質サマリも返す。"""
    quality = {
        "imputed_revenue_rows": 0,
        "dedup_groups": 0,
        "missing_counts": {},
        "negative_values": {},
        "outlier_counts": {},
    }

    # 必須列の存在チェック
    ok, missing = validate_schema(df)
    if not ok:
        raise ValueError(f"必須列が不足しています: {missing}")

    df = df.copy()

    # 型・日付
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["category"] = df["category"].astype("string")
    for c in ["units", "unit_price", "revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 欠損件数
    quality["missing_counts"] = df.isna().sum().to_dict()

    # revenue 補完（欠損/0/負）
    mask_revenue_bad = df["revenue"].isna() | (df["revenue"] <= 0)
    quality["imputed_revenue_rows"] = int(mask_revenue_bad.sum())
    df.loc[mask_revenue_bad, "revenue"] = df.loc[mask_revenue_bad, "units"] * df.loc[mask_revenue_bad, "unit_price"]

    # 重複集約キー
    key_cols = [
        "date",
        "category",
        "region",
        "sales_channel",
        "customer_segment",
        "unit_price",
    ]
    before = len(df)
    grouped = (
        df.groupby(key_cols, dropna=False)[["units", "revenue"]]
        .sum(min_count=1)
        .reset_index()
    )
    quality["dedup_groups"] = before - len(grouped)
    df = grouped

    # 時間列
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # 外れ値フラグ（IQR: unit_price, revenue）
    def iqr_flag(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.Series([False] * len(s), index=s.index)
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (s < lo) | (s > hi)

    df["outlier_unit_price"] = iqr_flag(df["unit_price"]) if df["unit_price"].notna().any() else False
    df["outlier_revenue"] = iqr_flag(df["revenue"]) if df["revenue"].notna().any() else False
    df["outlier_any"] = df[["outlier_unit_price", "outlier_revenue"]].any(axis=1)
    quality["outlier_counts"] = {
        "unit_price": int(df["outlier_unit_price"].sum()),
        "revenue": int(df["outlier_revenue"].sum()),
        "any": int(df["outlier_any"].sum()),
    }

    # ネガティブ値
    quality["negative_values"] = {
        "units": int((df["units"] < 0).sum()),
        "unit_price": int((df["unit_price"] < 0).sum()),
        "revenue": int((df["revenue"] < 0).sum()),
    }

    return df, quality


# -------------------------------------------------------------
# ビュー: サイドバー
# -------------------------------------------------------------

def sidebar_controls(df: pd.DataFrame, quality: dict) -> dict:
    st.sidebar.header("フィルタ & 設定")

    # 日付範囲
    min_date = pd.to_datetime(df["date"].min())
    max_date = pd.to_datetime(df["date"].max())
    d_start, d_end = st.sidebar.date_input(
        "期間",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        format="YYYY-MM-DD",
    )
    if isinstance(d_start, tuple):  # 古いStreamlit互換
        d_start, d_end = d_start

    # 多値フィルタ
    regions = sorted(df["region"].dropna().unique().tolist())
    channels = sorted(df["sales_channel"].dropna().unique().tolist())
    segments = sorted(df["customer_segment"].dropna().unique().tolist())

    sel_regions = st.sidebar.multiselect("地域", regions, default=regions)
    sel_channels = st.sidebar.multiselect("チャネル", channels, default=channels)
    sel_segments = st.sidebar.multiselect("セグメント", segments, default=segments)

    # 外れ値除外
    exclude_outliers = st.sidebar.toggle("外れ値を除外（IQR）", value=False)

    # 利益マージン
    margin = st.sidebar.slider("想定粗利率（%）", min_value=0, max_value=90, value=30, step=5)

    # 品質サマリ
    st.sidebar.divider()
    st.sidebar.subheader("データ品質サマリ")
    st.sidebar.write("**欠損件数**", quality.get("missing_counts", {}))
    st.sidebar.write("**外れ値件数**", quality.get("outlier_counts", {}))
    st.sidebar.write("**重複統合行数**", int(quality.get("dedup_groups", 0)))
    st.sidebar.write("**revenue補完件数**", int(quality.get("imputed_revenue_rows", 0)))

    return {
        "date_start": pd.to_datetime(d_start),
        "date_end": pd.to_datetime(d_end),
        "regions": sel_regions,
        "channels": sel_channels,
        "segments": sel_segments,
        "exclude_outliers": exclude_outliers,
        "margin": margin / 100.0,
    }


# -------------------------------------------------------------
# 集計関数
# -------------------------------------------------------------

def apply_filters(df: pd.DataFrame, ctrl: dict) -> pd.DataFrame:
    mask = (
        (df["date"].dt.date >= ctrl["date_start"].date())
        & (df["date"].dt.date <= ctrl["date_end"].date())
        & (df["region"].isin(ctrl["regions"]))
        & (df["sales_channel"].isin(ctrl["channels"]))
        & (df["customer_segment"].isin(ctrl["segments"]))
    )
    if ctrl["exclude_outliers"]:
        mask &= ~df["outlier_any"]
    return df.loc[mask].copy()


def kpis(df: pd.DataFrame, margin: float) -> dict:
    revenue_sum = float(df["revenue"].sum()) if len(df) else 0.0
    units_sum = float(df["units"].sum()) if len(df) else 0.0
    orders = int(len(df))
    wavg_unit_price = float((df["unit_price"] * df["units"]).sum() / units_sum) if units_sum > 0 else np.nan
    rev_per_order = (revenue_sum / orders) if orders > 0 else np.nan
    profit = revenue_sum * margin
    return {
        "revenue_sum": revenue_sum,
        "units_sum": units_sum,
        "orders": orders,
        "wavg_unit_price": wavg_unit_price,
        "rev_per_order": rev_per_order,
        "profit": profit,
    }


def ts_aggregate(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """freq='D' 日次, 'MS' 月初。
    戻り値: date列 + revenue
    """
    s = (
        df.set_index("date").sort_index()["revenue"].resample(freq).sum(min_count=1)
    )
    out = s.reset_index().rename(columns={"revenue": "売上"})
    return out


def add_yoy(df_ts: pd.DataFrame) -> pd.DataFrame:
    """月次時系列に前年同月値を付与（売上_prev）。"""
    df = df_ts.copy()
    df["売上_prev"] = df["売上"].shift(12)
    return df


# -------------------------------------------------------------
# ビュー: メイン
# -------------------------------------------------------------

def render_kpi_row(metrics: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("総売上", yen(metrics["revenue_sum"]))
    col2.metric("総数量", intfmt(metrics["units_sum"]))
    col3.metric("平均単価", yen(metrics["wavg_unit_price"]))
    col4.metric("客単価", yen(metrics["rev_per_order"]))
    col5.metric("推定利益", yen(metrics["profit"]))


def render_timeseries(df_filt: pd.DataFrame):
    st.subheader("売上推移（時系列）")
    ts_mode = st.radio("集計単位", ["日次", "月次"], horizontal=True, index=1)
    yoy = st.checkbox("前年同月を重ねて表示（※月次のみ）", value=True)

    if ts_mode == "日次":
        ts = ts_aggregate(df_filt, freq="D")
        base = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("date:T", title="日付"),
            y=alt.Y("売上:Q", title="売上"),
            tooltip=[alt.Tooltip("date:T", title="日付"), alt.Tooltip("売上:Q", title="売上", format=",.0f")],
        ).properties(height=320)
        st.altair_chart(base, use_container_width=True)
    else:
        ts = ts_aggregate(df_filt, freq="MS")
        ts = add_yoy(ts)
        line_curr = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("date:T", title="月"),
            y=alt.Y("売上:Q", title="売上"),
            color=alt.value("#1f77b4"),
            tooltip=[alt.Tooltip("date:T", title="月"), alt.Tooltip("売上:Q", title="売上", format=",.0f")],
        )
        chart = line_curr
        if yoy and ts["売上_prev"].notna().any():
            prev = ts.dropna(subset=["売上_prev"]).copy()
            prev["シリーズ"] = "前年同月"
            curr = ts.copy()
            curr["シリーズ"] = "当年"
            melt = pd.concat([
                curr[["date", "売上", "シリーズ"]].rename(columns={"売上": "値"}),
                prev[["date", "売上_prev", "シリーズ"]].rename(columns={"売上_prev": "値"}),
            ])
            chart = alt.Chart(melt).mark_line(point=True).encode(
                x=alt.X("date:T", title="月"),
                y=alt.Y("値:Q", title="売上"),
                color=alt.Color("シリーズ:N", title="シリーズ"),
                tooltip=[alt.Tooltip("date:T", title="月"), alt.Tooltip("値:Q", title="売上", format=",.0f"), alt.Tooltip("シリーズ:N")],
            )
        st.altair_chart(chart.properties(height=320), use_container_width=True)


def render_composition(df_filt: pd.DataFrame):
    st.subheader("構成比・カテゴリ/地域")
    left, right = st.columns(2)

    # カテゴリ別構成
    with left:
        mode = st.selectbox("カテゴリ構成の表示", ["ドーナツ", "棒グラフ"], index=0)
        cat = (
            df_filt.groupby("category", dropna=False)["revenue"].sum().reset_index().rename(columns={"revenue": "売上"})
        )
        cat["構成比"] = cat["売上"] / cat["売上"].sum() if cat["売上"].sum() else 0
        if mode == "ドーナツ":
            base = alt.Chart(cat).encode(
                theta=alt.Theta("売上:Q"),
                color=alt.Color("category:N", title="カテゴリ"),
                tooltip=["category:N", alt.Tooltip("売上:Q", format=",.0f"), alt.Tooltip("構成比:Q", format=".1%")],
            )
            chart = base.mark_arc(innerRadius=60)
        else:
            chart = alt.Chart(cat).mark_bar().encode(
                x=alt.X("売上:Q", title="売上"),
                y=alt.Y("category:N", title="カテゴリ", sort='-x'),
                tooltip=["category:N", alt.Tooltip("売上:Q", format=",.0f"), alt.Tooltip("構成比:Q", format=".1%")],
            )
        st.altair_chart(chart.properties(height=320), use_container_width=True)

    # 地域別売上
    with right:
        reg = (
            df_filt.groupby("region", dropna=False)["revenue"].sum().reset_index().rename(columns={"revenue": "売上"})
        )
        chart = alt.Chart(reg).mark_bar().encode(
            x=alt.X("売上:Q", title="売上"),
            y=alt.Y("region:N", title="地域", sort='-x'),
            tooltip=["region:N", alt.Tooltip("売上:Q", format=",.0f")],
        )
        st.altair_chart(chart.properties(height=320), use_container_width=True)


def render_crosstab(df_filt: pd.DataFrame):
    st.subheader("クロス分析")
    left, right = st.columns(2)

    with left:
        st.markdown("**セグメント × チャネル（売上）**")
        pvt = pd.pivot_table(
            df_filt,
            values="revenue",
            index="customer_segment",
            columns="sales_channel",
            aggfunc="sum",
            fill_value=0,
        )
        data = pvt.reset_index().melt("customer_segment", var_name="sales_channel", value_name="売上")
        chart = alt.Chart(data).mark_rect().encode(
            x=alt.X("sales_channel:N", title="チャネル"),
            y=alt.Y("customer_segment:N", title="セグメント"),
            color=alt.Color("売上:Q", title="売上", scale=alt.Scale(scheme="blues")),
            tooltip=["customer_segment:N", "sales_channel:N", alt.Tooltip("売上:Q", format=",.0f")],
        )
        st.altair_chart(chart.properties(height=320), use_container_width=True)

    with right:
        st.markdown("**カテゴリ × 地域（売上）**")
        pvt = pd.pivot_table(
            df_filt,
            values="revenue",
            index="category",
            columns="region",
            aggfunc="sum",
            fill_value=0,
        )
        data = pvt.reset_index().melt("category", var_name="region", value_name="売上")
        chart = alt.Chart(data).mark_rect().encode(
            x=alt.X("region:N", title="地域"),
            y=alt.Y("category:N", title="カテゴリ"),
            color=alt.Color("売上:Q", title="売上", scale=alt.Scale(scheme="greens")),
            tooltip=["category:N", "region:N", alt.Tooltip("売上:Q", format=",.0f")],
        )
        st.altair_chart(chart.properties(height=320), use_container_width=True)


def render_table_and_downloads(df_filt: pd.DataFrame):
    st.subheader("詳細テーブル")

    # 表示用フォーマット列
    show = df_filt.copy()
    show = show[
        [
            "date",
            "category",
            "units",
            "unit_price",
            "region",
            "sales_channel",
            "customer_segment",
            "revenue",
        ]
    ].sort_values("date")

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DatetimeColumn("日付", format="YYYY-MM-DD"),
            "category": st.column_config.TextColumn("カテゴリ"),
            "units": st.column_config.NumberColumn("数量", format=",d"),
            "unit_price": st.column_config.NumberColumn("単価", format="¥,d"),
            "region": st.column_config.TextColumn("地域"),
            "sales_channel": st.column_config.TextColumn("チャネル"),
            "customer_segment": st.column_config.TextColumn("セグメント"),
            "revenue": st.column_config.NumberColumn("売上", format="¥,d"),
        },
        height=360,
    )

    # ダウンロード
    st.markdown("#### ダウンロード")
    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSVをダウンロード", data=csv, file_name="filtered_sales.csv", mime="text/csv")

    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        parquet = io.BytesIO()
        show.to_parquet(parquet, index=False)
        st.download_button("Parquetをダウンロード", data=parquet.getvalue(), file_name="filtered_sales.parquet", mime="application/octet-stream")
    except Exception:
        st.caption("※ pyarrow が未インストールのため、Parquet出力は無効化されています。")


# -------------------------------------------------------------
# アプリ本体
# -------------------------------------------------------------

def main():
    st.title("📊 売上ダッシュボード（オリジナルBI）")
    st.caption("データ: data/sample_sales.csv")

    # 読み込み & 前処理
    try:
        raw = load_csv(DATA_PATH)
        df, quality = preprocess(raw)
    except Exception as e:
        st.error(f"データ読み込み/前処理でエラー: {e}")
        st.stop()

    # サイドバー
    ctrl = sidebar_controls(df, quality)

    # フィルタ適用
    df_filt = apply_filters(df, ctrl)

    # KPI
    st.divider()
    metrics = kpis(df_filt, margin=ctrl["margin"])
    render_kpi_row(metrics)

    # グラフ
    st.divider()
    render_timeseries(df_filt)

    st.divider()
    render_composition(df_filt)

    st.divider()
    render_crosstab(df_filt)

    # テーブル & DL
    st.divider()
    render_table_and_downloads(df_filt)

    # フッター的情報
    st.divider()
    total_raw = df["revenue"].sum()
    total_filtered = df_filt["revenue"].sum()
    st.caption(
        f"検証: 全体売上 = {yen(total_raw)} / フィルタ後 = {yen(total_filtered)} | 行数: 全体 {len(df):,} / フィルタ後 {len(df_filt):,}"
    )


if __name__ == "__main__":
    main()
