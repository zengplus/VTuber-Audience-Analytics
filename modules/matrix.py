import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
from id2name import id2name
import plotly.express as px

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def compute_matrix(_ids, conn):
    src = conn.execute(f"""
        SELECT DATE_TRUNC('month', month) AS month,
               source_liver, COUNT(*) cnt
        FROM new_source WHERE target_liver IN {tuple(_ids)}
        GROUP BY month, source_liver
    """).fetchdf()
    tgt = conn.execute(f"""
        SELECT DATE_TRUNC('month', month) AS month,
               target_liver, COUNT(*) cnt
        FROM new_target WHERE source_liver IN {tuple(_ids)}
        GROUP BY month, target_liver
    """).fetchdf()
    src["month"] = src["month"].dt.strftime("%Y-%m")
    tgt["month"] = tgt["month"].dt.strftime("%Y-%m")
    src["主播"] = src["source_liver"].map(id2name)
    tgt["主播"] = tgt["target_liver"].map(id2name)
    return src, tgt

@st.cache_data(show_spinner="计算中...")
def _cached_compute(_ids, conn):
    return compute_matrix(_ids, conn)

def show_matrix(T, conn):
    livers = list(id2name.keys())
    names = [id2name[i] for i in livers]
    sel_names = st.multiselect(T["select"], names, default=["嘉然"], key="m1")
    sel_ids = sorted([k for k, v in id2name.items() if v in sel_names])
    if not sel_ids:
        return

    cache_key = f"{'-'.join(map(str, sel_ids))}_{datetime.now():%Y-%m}"
    src_cache = CACHE_DIR / f"src_{cache_key}.parquet"
    tgt_cache = CACHE_DIR / f"tgt_{cache_key}.parquet"

    if st.button(T["reload"], key="m2"):
        src_cache.unlink(missing_ok=True)
        tgt_cache.unlink(missing_ok=True)
        st.cache_data.clear()

    if src_cache.exists() and tgt_cache.exists():
        src, tgt = pd.read_parquet(src_cache), pd.read_parquet(tgt_cache)
    else:
        src, tgt = _cached_compute(tuple(sel_ids), conn)
        src.to_parquet(src_cache, index=False)
        tgt.to_parquet(tgt_cache, index=False)

    src_tbl = src.pivot_table(index="month", columns="主播", values="cnt", fill_value=0).astype(int)
    st.subheader(T["src_table"])
    st.dataframe(src_tbl.style.background_gradient(cmap="YlGnBu"))
    st.subheader(T["src_heat"])
    st.plotly_chart(
        px.imshow(src_tbl, labels=dict(x="主播", y="月份", color="人数"),
                  color_continuous_scale="YlGnBu", aspect="auto"),
        use_container_width=True
    )

    tgt_tbl = tgt.pivot_table(index="month", columns="主播", values="cnt", fill_value=0).astype(int)
    st.subheader(T["tgt_table"])
    st.dataframe(tgt_tbl.style.background_gradient(cmap="YlGnBu"))
    st.subheader(T["tgt_heat"])
    st.plotly_chart(
        px.imshow(tgt_tbl, labels=dict(x="主播", y="月份", color="人数"),
                  color_continuous_scale="YlGnBu", aspect="auto"),
        use_container_width=True
    )