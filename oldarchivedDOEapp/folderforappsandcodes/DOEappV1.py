# %%
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import lhs, ff2n
import plotly.express as px
import plotly.graph_objects as go  # For contour plots

st.set_page_config(page_title="DOE-in-a-Box Pro", layout="wide")
st.title("DOE-in-a-Box Pro")
st.write("**Full control**: Custom ranges, DOE types, **real analysis**, contour plots, and **optimal blends**!")

# --- STEP 1: Factor Setup ---
st.subheader("1. Define Your Factors")
num_factors = st.slider("How many factors?", 2, 8, 3)

factor_data = []
for i in range(num_factors):
    with st.expander(f"Factor {i+1}", expanded=True):
        col1, col2 = st.columns([1, 1])
        name = col1.text_input("Name", f"Factor {i+1}", key=f"name_{i}")
        use_range = col2.checkbox("Use Range", value=True, key=f"range_{i}")
        
        if use_range:
            col_min, col_max = st.columns(2)
            min_val = col_min.number_input("Min", value=0.0, key=f"min_{i}")
            max_val = col_max.number_input("Max", value=100.0, key=f"max_{i}")
            factor_data.append({"name": name, "type": "range", "min": min_val, "max": max_val})
        else:
            fixed_val = st.number_input("Fixed Value", value=50.0, key=f"fixed_{i}")
            factor_data.append({"name": name, "type": "fixed", "value": fixed_val})

# --- STEP 2: DOE Type with Popups ---
st.subheader("2. Choose DOE Type")

doe_options = {
    "Latin Hypercube": "Best for **non-linear modeling** (RSM, ML). Space-filling. Flexible runs.",
    "Full Factorial": "All combinations. **Gold standard for interactions**. 2ⁿ runs.",
    "Fractional Factorial": "Screen many factors with **fewer runs**. Main effects + key interactions."
}

selected_doe = st.radio(
    "DOE Type",
    options=doe_options.keys(),
    format_func=lambda x: x,
    help="Click (i) for details"
)

with st.expander("Why this DOE?", expanded=False):
    st.info(doe_options[selected_doe])

# --- STEP 3: Runs ---
runs = st.number_input("Number of Runs", min_value=4, max_value=128, value=12)

# --- Generate DOE ---
if st.button("Generate DOE Plan", type="primary"):
    with st.spinner("Building your design..."):
        n = len(factor_data)
        
        if selected_doe == "Latin Hypercube":
            doe = lhs(n, samples=runs)
        elif selected_doe == "Full Factorial":
            base = ff2n(n)
            doe = np.tile(base, (int(np.ceil(runs / len(base))), 1))[:runs]
        else:  # Fractional Factorial
            base = ff2n(n)
            doe = base[:runs]  # Simple screening: first N runs

        # Apply scaling
        scaled = []
        for i, f in enumerate(factor_data):
            col = doe[:, i] if doe.ndim > 1 else doe
            if f["type"] == "fixed":
                scaled.append(np.full(runs, f["value"]))
            else:
                # Normalize FF from [-1,1] to [0,1]
                if selected_doe != "Latin Hypercube":
                    col = (col + 1) / 2
                scaled.append(col * (f["max"] - f["min"]) + f["min"])
        
        df = pd.DataFrame(np.column_stack(scaled).round(3), columns=[f["name"] for f in factor_data])
        
        # Auto-sum to 100% for mixtures
        if all("%" in name for name in df.columns):
            df = df.div(df.sum(axis=1), axis=0) * 100
            df = df.round(2)

        st.session_state.doe_plan = df
        st.success(f"{selected_doe} generated: {len(df)} runs")

# --- RESULTS ---
if 'doe_plan' in st.session_state:
    df = st.session_state.doe_plan
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode()
    st.download_button("Download DOE Plan", csv, "doe_plan.csv", "text/csv")

    if len(df.columns) >= 2:
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Design Space")
        st.plotly_chart(fig, use_container_width=True)

# --- ANALYSIS MODE ---
st.divider()
st.subheader("Next: Upload Results & Analyze")

if st.button("Enable Analysis Mode", type="secondary"):
    st.session_state.analysis = True

if st.session_state.get("analysis", False):
    st.info("Upload your **DOE results CSV** (must include your factor columns + response columns like Body, Tannin, Fruit)")

    uploaded = st.file_uploader("Upload Results CSV", type="csv")
    
    if uploaded and 'doe_plan' in st.session_state:
        try:
            results = pd.read_csv(uploaded)
            doe_plan = st.session_state.doe_plan
            
            merged = pd.merge(
                doe_plan.round(6),
                results.round(6),
                on=list(doe_plan.columns),
                how='inner'
            )
            
            if merged.empty:
                st.error("No matching runs found. Check column names and values.")
            else:
                st.success(f"Loaded {len(merged)} runs with responses!")
                st.dataframe(merged)

                response_col = st.selectbox("Select Response to Model", merged.columns[len(doe_plan.columns):])
                
                if st.button("Fit Model & Show Contour", type="primary"):
                    with st.spinner("Fitting quadratic model..."):
                        from sklearn.preprocessing import PolynomialFeatures
                        from sklearn.linear_model import LinearRegression
                        from sklearn.pipeline import make_pipeline

                        X = merged[doe_plan.columns]
                        y = merged[response_col]

                        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
                        model.fit(X, y)

                        x1 = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 30)
                        x2 = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 30)
                        X1, X2 = np.meshgrid(x1, x2)
                        grid = pd.DataFrame({X.columns[0]: X1.ravel(), X.columns[1]: X2.ravel()})
                        for col in X.columns[2:]:
                            grid[col] = X[col].mean()

                        Z = model.predict(grid).reshape(30, 30)

                        # CORRECT CONTOUR PLOT
                        fig = go.Figure(data=go.Contour(
                            x=x1, y=x2, z=Z,
                            colorscale='Viridis',
                            contours_coloring='lines',
                            line_width=1
                        ))
                        fig.update_layout(
                            title=f"{response_col} Response Surface",
                            xaxis_title=X.columns[0],
                            yaxis_title=X.columns[1]
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        best_idx = np.argmax(Z)
                        best_x1, best_x2 = X1.ravel()[best_idx], X2.ravel()[best_idx]
                        st.success(f"**Optimal Blend:** {X.columns[0]} = {best_x1:.2f}, {X.columns[1]} = {best_x2:.2f} → {response_col} = {Z.max():.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("First generate a DOE plan, then upload results with matching factor values.")

# %%



