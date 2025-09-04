import streamlit as st
import pandas as pd
from id2name import id2name

def show_trend(T, conn):
    sel = st.multiselect(T["select"], list(id2name.values()), default=["嘉然"], key="t1")
    sel_ids = [k for k, v in id2name.items() if v in sel]
    if not sel_ids:
        return
    df = conn.execute(f"""
        SELECT DATE_TRUNC('month', month) AS month,
               SUM(CASE WHEN target_liver IN {tuple(sel_ids)} THEN 1 ELSE 0 END) AS new_users,
               SUM(CASE WHEN source_liver IN {tuple(sel_ids)} THEN 1 ELSE 0 END) AS lost_users
        FROM new_source
        GROUP BY month
        ORDER BY month
    """).fetchdf()
    st.line_chart(df.set_index("month")[["new_users", "lost_users"]])