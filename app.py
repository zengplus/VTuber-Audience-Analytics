import streamlit as st
from pathlib import Path
import duckdb
from modules import show_matrix, show_trend, show_update, show_cluster, show_assoc
from id2name import id2name

DB_PATH = Path(__file__).with_name("mydb.duckdb")
@st.cache_resource
def get_conn():
    return duckdb.connect(str(DB_PATH))

# ------------------------- 语言包 -------------------------------
LANG = {
    "zh": {
        "lang_switch": "语言 / Language",
        "matrix": "用户流动矩阵分析",
        "trend": "月度用户流动趋势",
        "update": "用户记录增量更新",
        "cluster": "用户群体聚类分析",
        "assoc": "用户兴趣关联分析",
        "select": "选择主播",
        "reload": "重新计算",
        "src_table": "用户来源分析表",
        "tgt_table": "用户流失去向表",
        "src_heat": "来源热图",
        "tgt_heat": "去向热图",
        "start": "开始增量更新",
        "gen": "生成图表",
    },
    "en": {
        "lang_switch": "Language / 语言",
        "matrix": "User Transfer Matrix",
        "trend": "Monthly User Flow Trend",
        "update": "Incremental User Record Update",
        "cluster": "User Clustering",
        "assoc": "Interest Association",
        "select": "Select streamers",
        "reload": "Recalculate",
        "src_table": "User Source Table",
        "tgt_table": "User Target Table",
        "src_heat": "Source Heatmap",
        "tgt_heat": "Target Heatmap",
        "start": "Start incremental update",
        "gen": "Generate charts",
    },
}

# ------------------------- 侧边栏 -------------------------------
st.set_page_config(layout="wide")
with st.sidebar:
    lang = st.selectbox(LANG["zh"]["lang_switch"], ["zh", "en"],
                        format_func=lambda x: "中文" if x == "zh" else "English")
    T = LANG[lang]
    st.markdown("---")
    st.subheader("快速导航")
    for key, title in [
        ("matrix", T["matrix"]),
        ("trend", T["trend"]),
        ("cluster", T["cluster"]),
        ("assoc", T["assoc"]),
        ("update", T["update"]),
    ]:
        st.markdown(f"- <a href='#{key}' target='_self'>{title}</a>", unsafe_allow_html=True)

# ------------------------- 路由 -------------------------------
conn = get_conn()

st.markdown("<a name='matrix'></a>", unsafe_allow_html=True)
st.header(T["matrix"])
show_matrix(T, conn)

st.markdown("<a name='trend'></a>", unsafe_allow_html=True)
st.header(T["trend"])
show_trend(T, conn)

st.markdown("<a name='cluster'></a>", unsafe_allow_html=True)
st.header(T["cluster"])
show_cluster(T, conn)

st.markdown("<a name='assoc'></a>", unsafe_allow_html=True)
st.header(T["assoc"])
show_assoc(T, conn)

st.markdown("<a name='update'></a>", unsafe_allow_html=True)
st.header(T["update"])
show_update(T)
