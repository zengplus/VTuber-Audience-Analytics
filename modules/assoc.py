import streamlit as st
import pandas as pd
import plotly.express as px
from id2name import id2name

def _do_assoc(T, conn, target_ids, top_n, exclude_ylg):
    """真正执行关联饼图并展示的函数"""
    src_df = conn.execute(f"""
        SELECT source_liver AS liver, SUM(cnt) AS cnt
        FROM monthly_matrix
        WHERE target_liver IN {tuple(target_ids)}
        GROUP BY source_liver
    """).fetchdf()

    tgt_df = conn.execute(f"""
        SELECT target_liver AS liver, SUM(cnt) AS cnt
        FROM monthly_matrix_out
        WHERE source_liver IN {tuple(target_ids)}
        GROUP BY target_liver
    """).fetchdf()

    total_df = pd.concat([src_df, tgt_df]).groupby("liver", as_index=False)["cnt"].sum()
    total_df = total_df[~total_df["liver"].isin(target_ids)]
    if exclude_ylg:
        total_df = total_df[total_df["liver"] != -1]

    if total_df.empty:
        st.warning("No data")
        return

    top = total_df.sort_values("cnt", ascending=False).head(top_n)
    rest_cnt = total_df.iloc[top_n:]["cnt"].sum()
    if rest_cnt > 0:
        top = pd.concat([top, pd.DataFrame({"liver": [-999], "cnt": [rest_cnt]})])
        id2name[-999] = "Others"
    top["主播名"] = top["liver"].map(id2name)

    fig = px.pie(top, names="主播名", values="cnt",
                 title=f"Users who like {'/'.join([id2name[i] for i in target_ids])} also like",
                 color_discrete_sequence=px.colors.sequential.YlGnBu_r)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

def show_assoc(T, conn):
    livers = list(id2name.keys())
    names  = [id2name[i] for i in livers]

    # 预设
    default_targets = ["嘉然"] 
    default_top_n   = 8
    default_exclude = False

    with st.form("pie_form"):
        target_names = st.multiselect("Target streamers", names,
                                      default=default_targets, key="a1")
        target_ids   = [k for k, v in id2name.items() if v in target_names]

        exclude_ylg = st.checkbox("Exclude YLG (-1)", value=default_exclude)
        top_n       = st.slider("Top N in pie", 3, 15, default_top_n)
        submitted   = st.form_submit_button(T["gen"], use_container_width=True)

    # ---------- 第一次自动跑 ----------
    if "assoc_auto_run" not in st.session_state:
        _do_assoc(T, conn,
                  target_ids=[k for k, v in id2name.items() if v in default_targets],
                  top_n=default_top_n,
                  exclude_ylg=default_exclude)
        st.session_state.assoc_auto_run = True

    # ---------- 用户手动提交 ----------
    if submitted:
        _do_assoc(T, conn, target_ids, top_n, exclude_ylg)