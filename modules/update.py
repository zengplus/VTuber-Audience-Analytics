import streamlit as st
import pandas as pd
import requests
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "fans_events.parquet"
API_WATCH = "https://danmakus.com/api/v2/user/watchedChannels"

def show_update(T):
    if st.button(T["start"]):
        with st.spinner("拉取中..."):
            old = pd.read_parquet(DATA_PATH)
            uids = old["uid"].unique()
            rows = []
            for uid in uids:
                r = requests.get(API_WATCH, params={"uid": uid}, timeout=10)
                if r.status_code != 200:
                    continue
                for item in r.json().get("data", []):
                    rows.append({
                        "uid": uid,
                        "ts": pd.to_datetime(item["lastLiveDate"], unit="ms"),
                        "liver": int(item["uId"])
                    })
            new = pd.DataFrame(rows)
            new["key"] = new["uid"].astype(str) + "_" + new["ts"].astype(str) + "_" + new["liver"].astype(str)
            old["key"] = old["uid"].astype(str) + "_" + old["ts"].astype(str) + "_" + old["liver"].astype(str)
            new = new[~new["key"].isin(old["key"])].drop(columns="key")
            new = new.sort_values("ts").drop_duplicates(subset=["uid", "liver", new["ts"].dt.date])
            if not new.empty:
                pd.concat([old, new]).to_parquet(DATA_PATH, index=False)
                st.success(f"✅ 追加 {len(new)} 条记录")
            else:
                st.info("无新增记录")