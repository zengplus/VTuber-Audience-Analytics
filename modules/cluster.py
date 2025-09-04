import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from id2name import id2name

def _do_cluster(T, conn, sel_ids, max_users, n_clusters, exclude_ylg):
    """真正执行聚类并画图的函数，与之前一致，只是抽出来方便复用"""
    cond_liver = f"AND liver IN {tuple(sel_ids)}" if sel_ids else ""
    cond_ylg   = "AND liver != -1" if exclude_ylg else ""
    df = conn.execute(f"""
        SELECT uid, liver
        FROM events
        WHERE 1=1 {cond_liver} {cond_ylg}
    """).fetchdf()

    if df.empty:
        st.warning("无数据")
        return

    top_users = df["uid"].value_counts().head(max_users).index
    df = df[df["uid"].isin(top_users)]
    matrix = df.assign(flag=1).pivot_table(index="uid", columns="liver", values="flag", fill_value=0)

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        st.warning("数据过少")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(matrix)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)
    plot_df = pd.DataFrame(coords, columns=["x", "y"])
    plot_df["cluster"] = labels.astype(str)

    fig = px.scatter(plot_df, x="x", y="y", color="cluster",
                     title=f"{len(matrix)} users × {n_clusters} clusters")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top5 streamers per cluster")
    for c in sorted(plot_df["cluster"].unique()):
        idx = labels == int(c)
        top5 = matrix[idx].mean().sort_values(ascending=False).head(5)
        st.write(f"**Cluster {c}**: " + ", ".join([id2name[i] for i in top5.index]))

def show_cluster(T, conn):
    livers = list(id2name.keys())
    names  = [id2name[i] for i in livers]

    # 预设默认值
    default_sel      = [] # 默认空是全部主播 也可以输入"嘉然", "向晚", "贝拉"
    default_max      = 3000 # 默认聚 N 数量用户
    default_k        = 4  # 默认聚 N 类
    default_exclude  = True # 默认是否排除 YLG

    # 表单
    with st.form("cluster_form"):
        sel_names = st.multiselect(T["select"], names, default=default_sel, key="c1")
        sel_ids   = [k for k, v in id2name.items() if v in sel_names]

        total_users = conn.execute("""
            SELECT COUNT(DISTINCT uid) FROM events
        """).fetchone()[0]

        max_users   = st.slider("Max users to cluster", 100, int(total_users),
                                min(default_max, int(total_users)), step=100)
        n_clusters  = st.slider("Cluster count", 2, 10, default_k)
        exclude_ylg = st.checkbox("Exclude YLG (-1)", value=default_exclude)
        submitted   = st.form_submit_button(T["cluster"], use_container_width=True)

    # ---------- 第一次自动跑 ----------
    if "cluster_auto_run" not in st.session_state:
        _do_cluster(T, conn,
                    sel_ids=[k for k, v in id2name.items() if v in default_sel],
                    max_users=default_max,
                    n_clusters=default_k,
                    exclude_ylg=default_exclude)
        st.session_state.cluster_auto_run = True

    # ---------- 用户手动提交 ----------
    if submitted:
        _do_cluster(T, conn, sel_ids, max_users, n_clusters, exclude_ylg)