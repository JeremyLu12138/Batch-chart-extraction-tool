import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


FILE_PATH = Path("All_Rates_Comparison_final.xlsx")

CUSTOM_CSS = """
<style>
    :root {
        --bg: #0b1021;
        --panel: #11182b;
        --muted: #d0d7e2;
        --accent: #5ad8fa;
        --accent-2: #8c7bff;
        --border: #1d2942;
    }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    body { background-color: var(--bg); color: #f3f6ff; font-family: "Segoe UI","Helvetica Neue",Arial,sans-serif; }
    .stApp header { background: transparent; }
    section.main > div { background-color: var(--bg); }
    .glass {
        background: linear-gradient(135deg, rgba(17,24,43,0.85), rgba(17,24,43,0.7));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }
    h1, h2, h3, h4 { color: #f6f8ff; }
    .metric-title { color: var(--muted); font-size: 0.95rem; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f6f8ff; }
</style>
"""


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    sheets = pd.read_excel(path, sheet_name=None)
    frames = []

    for sheet_name, df in sheets.items():
        if df.empty:
            continue

        df = df.copy()
        rename_map = {
            "Benefit type": "BenefitType",
            "Benefit Type": "BenefitType",
            "Benefit": "BenefitType",
        }
        df = df.rename(columns=rename_map)

        # If BenefitType still missing, derive from sheet name (e.g., "IP" sheet)
        if "BenefitType" not in df.columns:
            df["BenefitType"] = sheet_name

        # Coerce Age
        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_company_columns(df: pd.DataFrame) -> list:
    base_cols = {
        "Age",
        "Gender",
        "Occupation",
        "Benefit",
        "BenefitType",
        "BenefitPeriod",
        "WaitingPeriod",
    }
    return [c for c in df.columns if c not in base_cols]


def build_lookup(df: pd.DataFrame, age: int, gender: str, occupation: str, benefit: str) -> pd.DataFrame:
    benefit_col = "BenefitType" if "BenefitType" in df.columns else None
    query = df.copy()
    query = query[query["Age"] == age]
    query = query[query["Gender"] == gender]
    query = query[query["Occupation"] == occupation]
    if benefit_col:
        query = query[query[benefit_col] == benefit]

    company_cols = get_company_columns(df)
    rows = []
    for comp in company_cols:
        rate = None
        if not query.empty:
            series = query[comp].dropna()
            if not series.empty:
                rate = series.iloc[0]
        rows.append({"Company": comp.replace(" rates", ""), "Rate": rate})
    result = pd.DataFrame(rows)
    result["Display"] = result["Rate"].apply(lambda x: "no available" if pd.isna(x) else x)
    return result[["Company", "Display"]]


def build_trend(df: pd.DataFrame, companies: list, gender: str, occupation: str, benefit: str,
                benefit_period: str | None = None, waiting_period: str | None = None) -> tuple[pd.DataFrame, list]:
    benefit_col = "BenefitType" if "BenefitType" in df.columns else None
    filtered = df.copy()
    filtered = filtered[filtered["Gender"] == gender]
    filtered = filtered[filtered["Occupation"] == occupation]
    if benefit_col:
        filtered = filtered[filtered[benefit_col] == benefit]
    if benefit_period and "BenefitPeriod" in filtered.columns:
        filtered = filtered[filtered["BenefitPeriod"] == benefit_period]
    if waiting_period and "WaitingPeriod" in filtered.columns:
        filtered = filtered[filtered["WaitingPeriod"] == waiting_period]

    melt_cols = [c for c in get_company_columns(df) if not companies or c in companies]
    if not melt_cols:
        return pd.DataFrame(), []

    missing = []
    for c in melt_cols:
        filtered[c] = pd.to_numeric(filtered[c], errors="coerce")
        if filtered[c].dropna().empty:
            missing.append(c)

    long_df = filtered[["Age"] + melt_cols].melt(id_vars=["Age"], var_name="Company", value_name="Rate")
    long_df = long_df.dropna(subset=["Rate"])
    long_df["Company"] = long_df["Company"].str.replace(" rates", "", regex=False)
    return long_df, missing


def build_relative_trend(df: pd.DataFrame, base_company: str, compare_companies: list,
                         gender: str, occupation: str, benefit: str,
                         benefit_period: str | None = None, waiting_period: str | None = None) -> tuple[pd.DataFrame, list, bool]:
    if not base_company or not compare_companies:
        return pd.DataFrame(), [], False

    benefit_col = "BenefitType" if "BenefitType" in df.columns else None
    filtered = df.copy()
    filtered = filtered[filtered["Gender"] == gender]
    filtered = filtered[filtered["Occupation"] == occupation]
    if benefit_col:
        filtered = filtered[filtered[benefit_col] == benefit]
    if benefit_period and "BenefitPeriod" in filtered.columns:
        filtered = filtered[filtered["BenefitPeriod"] == benefit_period]
    if waiting_period and "WaitingPeriod" in filtered.columns:
        filtered = filtered[filtered["WaitingPeriod"] == waiting_period]

    numeric_cols = [c for c in [base_company, *compare_companies] if c in filtered.columns]
    for c in numeric_cols:
        filtered[c] = pd.to_numeric(filtered[c], errors="coerce")

    base_series = (
        filtered[["Age", base_company]]
        .dropna(subset=["Age", base_company])
        .groupby("Age")[base_company]
        .mean()
    )
    baseline_missing = base_series.empty

    rows = []
    if not base_series.empty:
        base_df = base_series.reset_index()
        base_df["Company"] = base_company
        base_df["Ratio"] = 100.0
        rows.append(base_df[["Age", "Company", "Ratio"]])

    missing = []
    for comp in compare_companies:
        comp_series = (
            filtered[["Age", comp]]
            .dropna(subset=["Age", comp])
            .groupby("Age")[comp]
            .mean()
        )
        joined = pd.DataFrame({"base": base_series, "comp": comp_series}).dropna()
        if joined.empty:
            missing.append(comp)
            continue
        joined["Company"] = comp
        joined["Ratio"] = (joined["comp"] / joined["base"]) * 100
        rows.append(joined.reset_index()[["Age", "Company", "Ratio"]])

    if not rows:
        return pd.DataFrame(), missing, baseline_missing

    result = pd.concat(rows, ignore_index=True)
    result["Company"] = result["Company"].str.replace(" rates", "", regex=False)
    return result, missing, baseline_missing


def main():
    st.set_page_config(page_title="Rates Comparison Explorer", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Rates Comparison Explorer")
    st.caption(
        "Filter by demographic and benefit to inspect exact rates, then compare age-driven trends across companies, "
        "including baseline-relative views."
    )

    df = load_data(FILE_PATH)
    if df.empty:
        st.error(f"Data file not found: {FILE_PATH}")
        return

    company_cols = get_company_columns(df)
    if not company_cols:
        st.error("No company columns detected in the workbook.")
        return

    st.sidebar.header("Filters")
    age = st.sidebar.selectbox("Age", sorted(df["Age"].dropna().unique().tolist()))
    occupation = st.sidebar.selectbox("Occupation", sorted(df["Occupation"].dropna().unique().tolist()))
    gender = st.sidebar.selectbox("Gender", sorted(df["Gender"].dropna().unique().tolist()))

    benefits = df["BenefitType"].dropna().unique().tolist() if "BenefitType" in df.columns else []
    benefit = st.sidebar.selectbox("Benefit Type", sorted(benefits)) if benefits else ""

    ip_aliases = {"ip", "income protection"}
    is_ip = str(benefit).strip().lower() in ip_aliases
    benefit_period = ""
    waiting_period = ""
    if is_ip:
        bp_options = (
            sorted(df[df["BenefitType"] == benefit]["BenefitPeriod"].dropna().unique().tolist())
            if "BenefitPeriod" in df.columns else []
        )
        wp_options = (
            sorted(df[df["BenefitType"] == benefit]["WaitingPeriod"].dropna().unique().tolist())
            if "WaitingPeriod" in df.columns else []
        )
        benefit_period = st.sidebar.selectbox("Benefit Period", bp_options) if bp_options else ""
        waiting_period = st.sidebar.selectbox("Waiting Period", wp_options) if wp_options else ""

    st.markdown("### Rate Lookup")
    lookup_df = build_lookup(df, age, gender, occupation, benefit)
    st.dataframe(lookup_df, use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("### Trend by Age (per Company)")
    display_name = lambda c: c.replace(" rates", "")
    companies_selection = st.multiselect(
        "Select companies",
        options=company_cols,
        default=[],
        format_func=display_name,
    )

    if is_ip and (not benefit_period or not waiting_period):
        st.info("For IP, select Benefit Period and Waiting Period to view trends.")
    else:
        trend_df, missing_trend = build_trend(
            df,
            companies_selection,
            gender,
            occupation,
            benefit,
            benefit_period=benefit_period,
            waiting_period=waiting_period,
        )
        if not companies_selection:
            st.info("Select at least one company to display the trend.")
        elif trend_df.empty:
            st.info("No trend data available for the selected filters.")
        else:
            if missing_trend:
                st.warning(
                    "No rate data for selected combination in: "
                    + ", ".join(display_name(c) for c in missing_trend)
                )
            fig = px.line(
                trend_df,
                x="Age",
                y="Rate",
                color="Company",
                markers=True,
                title="Rate vs Age",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f6f8ff",
                legend_title_text="Company",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Relative Trend vs Baseline")
    st.caption(
        "Uses the same filters on the left (Age, Gender, Occupation, Benefit; Benefit Period / Waiting Period when Benefit is IP)."
    )

    base_company = st.selectbox("Baseline company", options=company_cols, format_func=display_name)
    compare_companies = st.multiselect(
        "Compare companies",
        options=[c for c in company_cols if c != base_company],
        default=[],
        format_func=display_name,
    )

    if is_ip and (not benefit_period or not waiting_period):
        st.info("For IP, select Benefit Period and Waiting Period to view the relative trend.")
    else:
        rel_trend_df, rel_missing, rel_base_missing = build_relative_trend(
            df,
            base_company,
            compare_companies,
            gender,
            occupation,
            benefit,
            benefit_period=benefit_period,
            waiting_period=waiting_period,
        )

        if not compare_companies:
            st.info("Please select at least one company to compare.")
        elif rel_base_missing:
            st.info(f"{display_name(base_company)} has no baseline rate for this combination.")
        elif rel_trend_df.empty:
            st.info("No relative trend data for this combination.")
        else:
            if rel_missing:
                st.warning(
                    "No rate data for this combination in: "
                    + ", ".join(display_name(c) for c in rel_missing)
                )
            fig_rel = px.line(
                rel_trend_df,
                x="Age",
                y="Ratio",
                color="Company",
                markers=True,
                title=f"Rate vs Age (% of {display_name(base_company)})",
                labels={"Ratio": "Rate (% of baseline)"},
            )
            fig_rel.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f6f8ff",
                legend_title_text="Company",
            )
            st.plotly_chart(fig_rel, use_container_width=True)


if __name__ == "__main__":
    main()
