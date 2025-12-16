import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Poverty & Millionaire Analytics", layout="wide")

st.title("Poverty & Millionaire Analytics Dashboard")
st.caption("Upload the Poverty/Millionaire dataset (Excel or CSV) to explore interactive insights.")

# ---------- Helpers ----------
def detect_population_col(columns):
    """
    Dataset sometimes has a typo like 'State Popiulation'.
    This function finds the population column robustly.
    """
    candidates = [
        "State Population",
        "State Popiulation",  # typo in your file
        "Population",
        "State Pop",
        "State Pop."
    ]
    for c in candidates:
        if c in columns:
            return c
    # fallback: try to find anything that looks like population
    for c in columns:
        if "pop" in c.lower():
            return c
    return None

def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # default to excel
        df = pd.read_excel(uploaded_file)

    # Basic cleanup
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------- Upload ----------
uploaded = st.file_uploader("Upload dataset (Excel .xlsx or CSV)", type=["xlsx", "xls", "csv"])
df = load_data(uploaded)

if df is None:
    st.info("Please upload the dataset to begin.")
    st.stop()

required_cols = {"State", "Number in Poverty", "Number of Millionaires"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    st.stop()

pop_col = detect_population_col(df.columns)
if pop_col is None:
    st.error("Could not find the population column (e.g., 'State Popiulation').")
    st.stop()

# Ensure numeric
for col in ["Number in Poverty", "Number of Millionaires", pop_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["State", "Number in Poverty", "Number of Millionaires", pop_col])

# Calculated fields (needed for Q2 + Q3)
df["Millionaire Density"] = df["Number of Millionaires"] / df[pop_col]
df["Poverty Rate"] = df["Number in Poverty"] / df[pop_col]

# ---------- Tabs (Q4) ----------
tab1, tab2, tab3 = st.tabs(["Poverty vs Millionaires", "Millionaire Density Map", "Poverty Rate"])

# =======================
# Q1: Poverty vs Millionaires
# =======================
with tab1:
    st.subheader("Q1: Compare Poverty vs Millionaire Population Across States")

    states = sorted(df["State"].astype(str).unique().tolist())
    default_states = states[:5] if len(states) >= 5 else states

    selected_states = st.multiselect(
        "Select at least 5 states",
        options=states,
        default=default_states
    )

    if len(selected_states) < 5:
        st.warning("Please select at least 5 states to meet the assignment requirement.")
    else:
        d1 = df[df["State"].isin(selected_states)].copy()

        # Side-by-side bar chart (Matplotlib)
        x = np.arange(len(d1))
        width = 0.4

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width/2, d1["Number in Poverty"], width, label="Number in Poverty")
        ax.bar(x + width/2, d1["Number of Millionaires"], width, label="Number of Millionaires")

        ax.set_xticks(x)
        ax.set_xticklabels(d1["State"])
        ax.set_xlabel("State")
        ax.set_ylabel("Population Count")
        ax.set_title("Poverty vs Millionaires (Selected States)")
        ax.legend()

        st.pyplot(fig, clear_figure=True)

        st.markdown("**Brief interpretation (edit as you like):**")
        st.write(
            "Across the selected states, the counts of people in poverty and the number of millionaires can vary widely. "
            "Some high-population states may show high values for both measures, while others may show relatively higher "
            "poverty compared to millionaire counts. This comparison helps illustrate how wealth and poverty can coexist "
            "differently across states, influenced by population size, cost of living, and regional economic conditions."
        )

# =======================
# Q2: Choropleth millionaire density map
# =======================
with tab2:
    st.subheader("Q2: Geographic Map of Millionaire Density by U.S. State")

    # Choropleth Map (Plotly Express)
    fig_map = px.choropleth(
        df,
        locations="State",            # expects 2-letter state codes for USA-states
        locationmode="USA-states",
        color="Millionaire Density",
        hover_name="State",
        hover_data={
            pop_col: True,
            "Number of Millionaires": True,
            "Millionaire Density": ":.4f"
        },
        scope="usa",
        title="Millionaire Density by State (Millionaires / Population)"
    )
    fig_map.update_layout(coloraxis_colorbar_title="Density")

    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("**Brief interpretation (4–6 sentences):**")
    st.write(
        "The choropleth highlights differences in millionaire concentration across the United States by normalizing "
        "millionaire counts by total state population. States with darker shading indicate a higher share of millionaires "
        "relative to population size. Patterns may show regional clustering where economic hubs and high-income areas are "
        "more prevalent. Meanwhile, lower-density states may reflect smaller high-net-worth populations or broader "
        "population bases. Overall, the map helps compare wealth concentration more fairly than raw counts alone."
    )

# =======================
# Q3: Poverty rate across states
# =======================
with tab3:
    st.subheader("Q3: Compare Poverty Rate Across States")

    d3 = df.sort_values("Poverty Rate", ascending=False).copy()

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.barh(d3["State"], d3["Poverty Rate"] * 100)
    ax2.invert_yaxis()

    ax2.set_xlabel("Poverty Rate (%)")
    ax2.set_ylabel("State")
    ax2.set_title("Poverty Rate by State (Highest to Lowest)")

    st.pyplot(fig2, clear_figure=True)

    st.markdown("**Brief interpretation (edit as you like):**")
    highest = d3.iloc[0]["State"]
    lowest = d3.iloc[-1]["State"]
    st.write(
        f"This chart ranks states by poverty rate (poverty ÷ population). "
        f"The highest poverty burden appears in **{highest}**, while the lowest appears in **{lowest}**. "
        "Looking at rates (not raw counts) helps identify which states are more heavily impacted relative to their size. "
        "These differences can reflect variations in wages, employment opportunities, cost of living, and social programs."
    )
