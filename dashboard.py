"""Streamlit dashboard wrapping the trained SchoolLSTM model.

Two pages:
  Browse  — explore precomputed predictions in inference_results.csv
  Predict — fill a feature form and run a single-row prediction

Run:
    streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from inference_lib import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    load_artifacts,
    predict_one,
)

INFERENCE_CSV = "inference_results.csv"
LATEST_DATA_CSV = "training-data/2024-data.csv"

st.set_page_config(page_title="School pct_ccr predictor", layout="wide")


@st.cache_resource(show_spinner="Loading model and encoders...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Loading predictions...")
def get_inference_results() -> pd.DataFrame:
    df = pd.read_csv(INFERENCE_CSV, dtype={"school_code": str, "grade": str, "subgroup": str})
    return df


@st.cache_data(show_spinner="Loading reference data...")
def get_latest_data() -> pd.DataFrame:
    if not Path(LATEST_DATA_CSV).exists():
        return pd.DataFrame()
    df = pd.read_csv(LATEST_DATA_CSV, low_memory=False)
    df["school_code"] = df["school_code"].astype(str)
    return df


@st.cache_data
def get_school_index(predictions: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    cols = ["school_code"]
    if "name" in latest.columns:
        cols.append("name")
    if not latest.empty:
        idx = latest[cols].drop_duplicates("school_code").sort_values("school_code")
    else:
        idx = predictions[["school_code"]].drop_duplicates().sort_values("school_code")
        idx["name"] = ""
    return idx.reset_index(drop=True)


def school_label(row: pd.Series) -> str:
    name = row.get("name", "")
    if isinstance(name, str) and name.strip():
        return f"{row['school_code']} — {name}"
    return row["school_code"]


def latest_row_for(latest: pd.DataFrame, school_code: str) -> dict | None:
    if latest.empty:
        return None
    rows = latest[latest["school_code"] == school_code]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def render_disclaimer() -> None:
    st.warning(
        "**Calibration note** — this dashboard surfaces raw model output. "
        "Current residuals on `inference_results.csv` are large (predictions can swing "
        "wildly from actuals). Treat results as illustrative; the model itself needs "
        "further tuning before operational use. See README/plan for details."
    )


def page_browse(predictions: pd.DataFrame) -> None:
    st.header("Browse schools and predictions")
    st.caption(f"{len(predictions):,} prediction rows from `{INFERENCE_CSV}`")

    with st.sidebar:
        st.subheader("Filters")
        code_query = st.text_input("School code contains", "")
        grades = sorted(predictions["grade"].dropna().unique().tolist())
        subgroups = sorted(predictions["subgroup"].dropna().unique().tolist())
        sel_grades = st.multiselect("Grade", grades)
        sel_subgroups = st.multiselect("Subgroup", subgroups)
        min_resid = float(predictions["residual"].min())
        max_resid = float(predictions["residual"].max())
        resid_range = st.slider(
            "Residual range (actual - predicted)",
            min_value=float(np.floor(min_resid)),
            max_value=float(np.ceil(max_resid)),
            value=(float(np.floor(min_resid)), float(np.ceil(max_resid))),
        )

    df = predictions
    if code_query:
        df = df[df["school_code"].str.contains(code_query, case=False, na=False)]
    if sel_grades:
        df = df[df["grade"].isin(sel_grades)]
    if sel_subgroups:
        df = df[df["subgroup"].isin(sel_subgroups)]
    df = df[(df["residual"] >= resid_range[0]) & (df["residual"] <= resid_range[1])]

    if df.empty:
        st.info("No rows match the current filters.")
        return

    mae = df["residual"].abs().mean()
    rmse = float(np.sqrt((df["residual"] ** 2).mean()))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Mean residual", f"{df['residual'].mean():.2f}")
    c3.metric("MAE", f"{mae:.2f}")
    c4.metric("RMSE", f"{rmse:.2f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Filtered predictions")
        st.dataframe(df, width="stretch", height=520)
    with right:
        st.subheader("Actual vs predicted")
        sample = df.sample(min(len(df), 5000), random_state=0)
        chart_df = sample.rename(
            columns={"actual_pct_ccr": "actual", "predicted_pct_ccr": "predicted"}
        )[["actual", "predicted"]]
        st.scatter_chart(chart_df, x="actual", y="predicted")
        st.caption("Perfect predictions land on the y=x diagonal.")


def page_predict(
    predictions: pd.DataFrame,
    latest: pd.DataFrame,
    artifacts: dict,
) -> None:
    st.header("Predict pct_ccr for a single cohort")
    encoders = artifacts["encoders"]

    school_index = get_school_index(predictions, latest)
    school_options = school_index.apply(school_label, axis=1).tolist()
    school_codes = school_index["school_code"].tolist()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Cohort")
        choice = st.selectbox(
            "School", options=range(len(school_options)),
            format_func=lambda i: school_options[i] if i < len(school_options) else "",
        )
        school_code = school_codes[choice]

        grade = st.selectbox("Grade", encoders.get("grade", []) or ["ALL"])
        subgroup = st.selectbox("Subgroup", encoders.get("subgroup", []) or ["ALL"])
        subject = st.selectbox("Subject", encoders.get("subject", []) or ["ALL"])

        defaults = latest_row_for(latest, school_code) or {}
        county_opts = encoders.get("county", []) or ["Not Found"]
        city_opts = encoders.get("city", []) or ["not found"]

        def _safe_index(opts: list[str], val) -> int:
            s = "" if val is None else str(val)
            return opts.index(s) if s in opts else 0

        county = st.selectbox(
            "County", county_opts, index=_safe_index(county_opts, defaults.get("county"))
        )
        city = st.selectbox(
            "City", city_opts, index=_safe_index(city_opts, defaults.get("city"))
        )

    with col_right:
        st.subheader("Numeric features")
        st.caption(
            "Defaults are pulled from the school's most recent row in "
            "`training-data/2024-data.csv` when available."
        )

        def _num_default(key: str, fallback: float) -> float:
            v = defaults.get(key, None)
            try:
                if v is None or pd.isna(v):
                    return float(fallback)
                return float(v)
            except (TypeError, ValueError):
                return float(fallback)

        year = st.number_input("year", min_value=2000, max_value=2100, value=2024, step=1)
        ratio = st.number_input("ratio (teacher/student)", value=_num_default("ratio", 15.0))
        violent = st.number_input("Violent Crime Total", value=_num_default("Violent Crime Total", 0.0))
        prop = st.number_input("Property Crime Total", value=_num_default("Property Crime Total", 0.0))
        burg = st.number_input("Burglary", value=_num_default("Burglary", 0.0))
        larc = st.number_input("Larceny-theft", value=_num_default("Larceny-theft", 0.0))
        pct_np = st.number_input("pct_notprof", value=_num_default("pct_notprof", 0.0))
        mvt = st.number_input("Motor vehicle theft", value=_num_default("Motor vehicle theft", 0.0))

    st.divider()
    if st.button("Predict", type="primary", width="stretch"):
        form = {
            "year": year, "ratio": ratio,
            "Violent Crime Total": violent, "Property Crime Total": prop,
            "Burglary": burg, "Larceny-theft": larc,
            "pct_notprof": pct_np, "Motor vehicle theft": mvt,
            "grade": grade, "subgroup": subgroup,
            "county": county, "city": city, "subject": subject,
        }
        pred = predict_one(form, artifacts)

        match = predictions[
            (predictions["school_code"] == school_code)
            & (predictions["grade"].astype(str) == str(grade))
            & (predictions["subgroup"].astype(str) == str(subgroup))
        ]
        actual = float(match["actual_pct_ccr"].iloc[0]) if not match.empty else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted pct_ccr", f"{pred:.2f}")
        if actual is not None:
            c2.metric("Actual pct_ccr (2024)", f"{actual:.2f}")
            c3.metric("Residual", f"{actual - pred:.2f}")
        else:
            c2.metric("Actual pct_ccr (2024)", "—")
            c3.metric("Residual", "—")

        with st.expander("Form values used"):
            st.json(form)

    st.caption(
        "Sequence length: 1 timestep (matches `streaming.py`). The model was trained on "
        "3-year sequences, so single-step predictions may underperform training MAE."
    )


def main() -> None:
    st.title("School pct_ccr predictor")
    render_disclaimer()

    artifacts = get_artifacts()
    predictions = get_inference_results()
    latest = get_latest_data()

    page = st.sidebar.radio("Page", ["Browse", "Predict"])
    if page == "Browse":
        page_browse(predictions)
    else:
        page_predict(predictions, latest, artifacts)


if __name__ == "__main__":
    main()
