#!/usr/bin/env python3
# streamlit run app.py
import streamlit as st
import pandas as pd
import duckdb, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from id2name import id2name
# 放在文件顶部，与其他 import 放一起
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------ 常量 ---------------------------------
DB_PATH   = Path(__file__).with_name("mydb.duckdb")
DATA_PATH = Path(__file__).with_name("data") / "fans_events.parquet"
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(exist_ok=True)

@st.cache_resource
def get_conn():
    return duckdb.connect(str(DB_PATH))

# ------------------------- 语言包 -------------------------------
LANG = {
    "zh": {
        "lang_switch": "语言 / Language",
        "matrix": "用户流动矩阵分析",
        "trend": "用户流动趋势分析",
        "aarrr": "AARRR 漏斗分析",
        "rfm": "RFM 用户分层",
        "cluster": "用户群体聚类分析",
        "assoc": "用户兴趣关联分析",
        "update": "用户记录增量更新",
        "select": "选择主播",
        "reload": "重新计算",
        "src_table": "用户来源分析表",
        "tgt_table": "用户流失去向表",
        "src_heat": "来源热图",
        "tgt_heat": "去向热图",
        "start": "开始增量更新",
        "gen": "生成图表",
        "funnel_acquisition": "Acquisition 月新增",
        "funnel_activation": "Activation 当月活跃",
        "funnel_retention": "Retention 次月回流",
        "funnel_revenue": "Revenue 付费人数",
        "funnel_referral": "Referral 推荐人数",
        "rfm_score": "RFM 得分分布",
        "rfm_segment": "RFM 分层结果",
    },
    "en": {
        "lang_switch": "Language / 语言",
        "matrix": "User Transfer Matrix",
        "trend": "User Flow Trend Analysis",
        "aarrr": "AARRR Funnel",
        "rfm": "RFM Segmentation",
        "cluster": "User Clustering",
        "assoc": "Interest Association",
        "update": "Incremental Update",
        "select": "Select streamers",
        "reload": "Recalculate",
        "src_table": "User Source Table",
        "tgt_table": "User Target Table",
        "src_heat": "Source Heatmap",
        "tgt_heat": "Target Heatmap",
        "start": "Start incremental update",
        "gen": "Generate",
        "funnel_acquisition": "Acquisition (New)",
        "funnel_activation": "Activation (Active)",
        "funnel_retention": "Retention (Return)",
        "funnel_revenue": "Revenue (Pay)",
        "funnel_referral": "Referral (Invite)",
        "rfm_score": "RFM Score",
        "rfm_segment": "RFM Segment",
    },
}

# ------------------------------ 语言切换 -------------------------------
language = st.sidebar.selectbox("Language", ["zh", "en"], format_func=lambda x: LANG[x]["lang_switch"])
T = LANG[language]

# 提前把 livers / names 准备好，后面所有模块都能用
livers = list(id2name.keys())
names  = [id2name[i] for i in livers]

# ------------------------------ 首页 -------------------------------
st.set_page_config(layout="wide") # 把页面换到宽体，使页面变更大
st.title("VTuber Flow and Audience Analytics System")
st.markdown("---")


# ------------------------------ 1. 月度趋势 -------------------------------
st.header(T["trend"])
sel_trend = st.multiselect(T["select"], names, default=["嘉然"], key="trend_sel")
sel_ids_t = [k for k, v in id2name.items() if v in sel_trend]
if sel_ids_t:
    conn = get_conn()
    df_trend = conn.execute(f"""
        SELECT DATE_TRUNC('month', month) AS month,
               SUM(CASE WHEN target_liver IN {tuple(sel_ids_t)} THEN 1 ELSE 0 END) AS new_users,
               SUM(CASE WHEN source_liver IN {tuple(sel_ids_t)} THEN 1 ELSE 0 END) AS lost_users
        FROM new_source
        GROUP BY month ORDER BY month
    """).fetchdf()
    st.line_chart(df_trend.set_index("month")[["new_users", "lost_users"]])


# ------------------------------ 2. 用户流动矩阵 -------------------------------
st.header(T["matrix"])
sel_names = st.multiselect(T["select"], names, default=["嘉然"], key="matrix_sel")
sel_ids   = [k for k, v in id2name.items() if v in sel_names]
if sel_ids:
    cache_key = f"{'-'.join(map(str, sel_ids))}_{datetime.now():%Y-%m}"
    src_cache = CACHE_DIR / f"src_{cache_key}.parquet"
    tgt_cache = CACHE_DIR / f"tgt_{cache_key}.parquet"

    if st.button(T["reload"], key="matrix_reload"):
        src_cache.unlink(missing_ok=True); tgt_cache.unlink(missing_ok=True)

    @st.cache_data(show_spinner=True)
    def compute_matrix(_ids):
        conn = get_conn()
        src = conn.execute(f"""
            SELECT DATE_TRUNC('month', month) AS month, source_liver, COUNT(*) cnt
            FROM new_source WHERE target_liver IN {tuple(_ids)}
            GROUP BY month, source_liver
        """).fetchdf()
        tgt = conn.execute(f"""
            SELECT DATE_TRUNC('month', month) AS month, target_liver, COUNT(*) cnt
            FROM new_target WHERE source_liver IN {tuple(_ids)}
            GROUP BY month, target_liver
        """).fetchdf()
        src["month"] = src["month"].dt.strftime("%Y-%m")
        tgt["month"] = tgt["month"].dt.strftime("%Y-%m")
        src["主播"] = src["source_liver"].map(id2name)
        tgt["主播"] = tgt["target_liver"].map(id2name)
        src.to_parquet(src_cache, index=False)
        tgt.to_parquet(tgt_cache, index=False)
        return src, tgt

    if src_cache.exists() and tgt_cache.exists():
        src, tgt = pd.read_parquet(src_cache), pd.read_parquet(tgt_cache)
    else:
        src, tgt = compute_matrix(sel_ids)

    src_tbl = src.pivot_table(index="month", columns="主播", values="cnt", fill_value=0).astype(int)
    tgt_tbl = tgt.pivot_table(index="month", columns="主播", values="cnt", fill_value=0).astype(int)

    # —— 来源部分 ——
    st.subheader(T["src_table"])
    st.dataframe(src_tbl.style.background_gradient(cmap="YlGnBu"))
    st.subheader(T["src_heat"])
    st.plotly_chart(px.imshow(src_tbl, labels=dict(x="主播", y="月份", color="人数"),
                                color_continuous_scale="YlGnBu", aspect="auto"), use_container_width=True)

    # —— 去向部分 ——
    st.subheader(T["tgt_table"])
    st.dataframe(tgt_tbl.style.background_gradient(cmap="YlGnBu"))
    st.subheader(T["tgt_heat"])
    st.plotly_chart(px.imshow(tgt_tbl, labels=dict(x="主播", y="月份", color="人数"),
                                color_continuous_scale="YlGnBu", aspect="auto"), use_container_width=True)

# ------------------------------ 3. AARRR 漏斗 -------------------------------
st.header(T["aarrr"])
conn = get_conn()
funnel_month = st.selectbox("Select month", sorted(conn.execute("SELECT DISTINCT month FROM events").fetchdf()["month"].dt.strftime("%Y-%m")))

def AARRR(month: str):
    m0 = datetime.strptime(month, "%Y-%m")
    m1 = (m0.replace(day=1) + timedelta(days=32)).replace(day=1)
    acquisition = conn.execute("""
        SELECT COUNT(DISTINCT uid) FROM liver_new WHERE month=?
    """, [m0]).fetchone()[0]
    activation = conn.execute("""
        SELECT COUNT(DISTINCT uid) FROM events WHERE ts>=? AND ts<?
    """, [m0, m1]).fetchone()[0]
    retention = conn.execute("""
        SELECT COUNT(DISTINCT uid) FROM events WHERE ts>=? AND ts<?
        AND uid IN (SELECT uid FROM liver_new WHERE month=?)
    """, [m1, (m1 + timedelta(days=32)).replace(day=1), m0]).fetchone()[0]
    # 简化：付费=Revenue，推荐=Referral 均用 0 占位，后续可接真实表
    revenue = 0; referral = 0
    return dict(acq=acquisition, act=activation, ret=retention, rev=revenue, ref=referral)

funnel = AARRR(funnel_month)
fig_f = go.Figure(go.Funnel(
    y=[T["funnel_acquisition"], T["funnel_activation"], T["funnel_retention"],
       T["funnel_revenue"], T["funnel_referral"]],
    x=[funnel["acq"], funnel["act"], funnel["ret"], funnel["rev"], funnel["ref"]],
    textinfo="value+percent initial"))
st.plotly_chart(fig_f, use_container_width=True)

# ------------------------------ 4. RFM -------------------------------
st.header(T["rfm"])
@st.cache_data
def rfm_now():
    conn = get_conn()
    return conn.execute("""
        SELECT uid,
               liver,
               recent_days,
               behavior_freq  AS frequency,
               monetary_value,
               r_score,
               f_score,
               m_score,
               rfm_code,
               rfm_user_tag
        FROM rfm_user
    """).fetchdf()

# 后面直接用这个 df，别再自己算 r/f/m
rfm_df = rfm_now()
col1, col2 = st.columns(2)
with col1:
    st.subheader(T["rfm_score"])
    st.bar_chart(rfm_df["rfm_code"].value_counts().sort_index())
with col2:
    st.subheader(T["rfm_segment"])
    st.bar_chart(rfm_df["rfm_user_tag"].value_counts())

# ------------------------------ 5. 聚类 -------------------------------
st.header(T["cluster"])

livers = list(id2name.keys())
names  = [id2name[i] for i in livers]

# 默认参数
DEFAULT_SEL     = []          # 不限主播
DEFAULT_MAX_U   = 3000
DEFAULT_K       = 4
DEFAULT_EXCLUDE = True

# 真正计算：交给 @st.cache_data 缓存
@st.cache_data(show_spinner=True)
def _compute_cluster(_ids: tuple, max_u: int, k: int, exclude: bool):
    """
    返回 (plot_df, labels, matrix)
    数据不足返回 (None,None,None)
    注意：_ids 用 tuple 才能被哈希
    """
    conn = get_conn()
    cond_liver = f"AND liver IN {_ids}" if _ids else ""
    cond_ylg   = "AND liver != -1" if exclude else ""
    df = conn.execute(f"""
        SELECT uid, liver
        FROM events
        WHERE 1=1 {cond_liver} {cond_ylg}
    """).fetchdf()

    if df.empty:
        return None, None, None

    top_u = df["uid"].value_counts().head(max_u).index
    df    = df[df["uid"].isin(top_u)]
    matrix = df.assign(flag=1).pivot_table(index="uid", columns="liver", values="flag", fill_value=0)

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None, None, None

    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(matrix)
    pca    = PCA(2, random_state=42).fit_transform(matrix)
    plot_df = pd.DataFrame(pca, columns=["x", "y"])
    plot_df["cluster"] = labels.astype(str)
    return plot_df, labels, matrix

def _show_cluster(plot_df, labels, matrix):
    fig = px.scatter(plot_df, x="x", y="y", color="cluster",
                     title=f"{len(matrix)} users × {len(plot_df['cluster'].unique())} clusters")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top5 streamers per cluster")
    for c in sorted(plot_df["cluster"].unique()):
        idx  = labels == int(c)
        top5 = matrix[idx].mean().sort_values(ascending=False).head(5)
        st.write(f"**Cluster {c}**: " + ", ".join([id2name[i] for i in top5.index]))

# 参数面板
with st.form("cluster_form"):
    sel_names = st.multiselect(T["select"], names, default=DEFAULT_SEL, key="c1")
    sel_ids   = tuple([k for k, v in id2name.items() if v in sel_names])  # 转 tuple 以便哈希

    total_u = get_conn().execute("SELECT COUNT(DISTINCT uid) FROM events").fetchone()[0]
    max_u   = st.slider("Max users", 100, int(total_u), min(DEFAULT_MAX_U, int(total_u)), 100)
    k       = st.slider("Cluster count", 2, 10, DEFAULT_K)
    exclude = st.checkbox("Exclude YLG", value=DEFAULT_EXCLUDE)
    run     = st.form_submit_button(T["gen"], use_container_width=True)

# 清缓存按钮（可选）
if st.button("🗑 清聚类缓存"):
    _compute_cluster.clear()   # 手动失效
    st.success("缓存已清空，请重新生成！")

# 首次自动跑
if "cluster_auto" not in st.session_state:
    plot_df, labels, matrix = _compute_cluster(tuple(DEFAULT_SEL), DEFAULT_MAX_U, DEFAULT_K, DEFAULT_EXCLUDE)
    if plot_df is not None:
        _show_cluster(plot_df, labels, matrix)
    st.session_state.cluster_auto = True

# 手动提交
if run:
    plot_df, labels, matrix = _compute_cluster(sel_ids, max_u, k, exclude)
    if plot_df is None:
        st.warning("数据过少或无数据")
    else:
        _show_cluster(plot_df, labels, matrix)


# ------------------------------ 6. 兴趣关联 -------------------------------
st.header(T["assoc"])

livers = list(id2name.keys())
names  = [id2name[i] for i in livers]

# 默认参数
DEFAULT_TARGETS = ["嘉然"]
DEFAULT_TOP_N   = 8
DEFAULT_EXCLUDE = False

# 真正计算：@st.cache_data 托管
@st.cache_data(show_spinner=True)
def _compute_assoc(target_ids: tuple, top_n: int, exclude: bool):
    """
    返回 top_df（含主播名列），不足返回 None
    """
    conn = get_conn()
    src_df = conn.execute(f"""
        SELECT source_liver AS liver, SUM(cnt) AS cnt
        FROM monthly_matrix
        WHERE target_liver IN {target_ids}
        GROUP BY source_liver
    """).fetchdf()

    tgt_df = conn.execute(f"""
        SELECT target_liver AS liver, SUM(cnt) AS cnt
        FROM monthly_matrix_out
        WHERE source_liver IN {target_ids}
        GROUP BY target_liver
    """).fetchdf()

    total_df = pd.concat([src_df, tgt_df]).groupby("liver", as_index=False)["cnt"].sum()
    total_df = total_df[~total_df["liver"].isin(target_ids)]
    if exclude:
        total_df = total_df[total_df["liver"] != -1]

    if total_df.empty:
        return None

    top = total_df.sort_values("cnt", ascending=False).head(top_n)
    rest_cnt = total_df.iloc[top_n:]["cnt"].sum()
    if rest_cnt > 0:
        top = pd.concat([top, pd.DataFrame({"liver": [-999], "cnt": [rest_cnt]})])
        id2name[-999] = "Others"
    top["主播名"] = top["liver"].map(id2name)
    return top

def _show_assoc(top_df):
    fig = px.pie(top_df, names="主播名", values="cnt",
                 title=f"Users who like {'/'.join(top_df.loc[top_df['liver']!=-999, '主播名'])} also like",
                 color_discrete_sequence=px.colors.sequential.YlGnBu_r)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

# 参数面板
with st.form("assoc_form"):
    target_names = st.multiselect("Target streamers", names,
                                  default=DEFAULT_TARGETS, key="a1")
    target_ids = tuple([k for k, v in id2name.items() if v in target_names])  # 转 tuple 供哈希

    exclude = st.checkbox("Exclude YLG (-1)", value=DEFAULT_EXCLUDE)
    top_n   = st.slider("Top N in pie", 3, 15, DEFAULT_TOP_N)
    run     = st.form_submit_button(T["gen"], use_container_width=True)

# 清缓存按钮（可选）
if st.button("🗑 清兴趣关联缓存"):
    _compute_assoc.clear()
    st.success("缓存已清空，请重新生成！")

# 首次自动跑
if "assoc_auto" not in st.session_state:
    top_df = _compute_assoc(tuple([k for k, v in id2name.items() if v in DEFAULT_TARGETS]),
                            DEFAULT_TOP_N, DEFAULT_EXCLUDE)
    if top_df is not None:
        _show_assoc(top_df)
    st.session_state.assoc_auto = True

# 手动提交
if run:
    top_df = _compute_assoc(target_ids, top_n, exclude)
    if top_df is None:
        st.warning("No data")
    else:
        _show_assoc(top_df)

# ------------------------------ 7. 增量更新 -------------------------------
st.header(T["update"])
if st.button(T["start"], key="update_btn"):
    with st.spinner("Pulling..."):
        old = pd.read_parquet(DATA_PATH)
        uids = old["uid"].unique()
        rows = []
        for uid in uids:
            r = requests.get("https://danmakus.com/api/v2/user/watchedChannels", params={"uid": uid}, timeout=10)
            if r.status_code != 200: continue
            for item in r.json().get("data", []):
                rows.append({"uid": uid, "ts": pd.to_datetime(item["lastLiveDate"], unit="ms"), "liver": int(item["uId"])})
        new = pd.DataFrame(rows)
        new["key"] = new["uid"].astype(str) + "_" + new["ts"].astype(str) + "_" + new["liver"].astype(str)
        old["key"] = old["uid"].astype(str) + "_" + old["ts"].astype(str) + "_" + old["liver"].astype(str)
        new = new[~new["key"].isin(old["key"])].drop(columns="key")
        if not new.empty:
            pd.concat([old, new]).to_parquet(DATA_PATH, index=False)
            st.success(f"✅ Added {len(new)} rows")
        else:
            st.info("No new rows")

