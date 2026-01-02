#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
import numpy as np
from pyDOE3 import lhs, ff2n, pbdesign
import plotly.express as px
import plotly.graph_objects as go
import base64
import statsmodels.api as sm
from statsmodels.formula.api import ols
import re

st.set_page_config(page_title="Pocket DOE", layout="wide")

# --- Background Image (from repo — safe for deployment) ---
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Image must be uploaded to repo as background.png
image_base64 = get_base64_image("background.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255, 255, 255, 0.78);
        z-index: -1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Pocket DOE")
st.write("**Smart experiments always at hand**")

# --- STEP 1: Factor Setup ---
st.subheader("1. Define Your Factors")
num_factors = st.slider("How many factors?", 2, 10, 3)

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

# --- STEP 2: DOE Type ---
st.subheader("2. Choose DOE Type")
doe_options = {
    "Full Factorial": "All combinations — gold standard for interactions.",
    "Fractional Factorial": "Efficient screening — flexible run count.",
    "Plackett-Burman": "Ultra-efficient main effects screening.",
    "Latin Hypercube": "Space-filling — ideal for modeling & exploration.",
    "Mixture Design": "Blends summing to 100% — space-filling LHS in simplex.",
    "Combined Mixture-Process": "Mixtures + process variables (temp, pH, etc.)."
}

selected_doe = st.radio("DOE Type", options=doe_options.keys())
with st.expander("Why this DOE?", expanded=False):
    st.info(doe_options[selected_doe])

is_mixture = selected_doe in ["Mixture Design", "Combined Mixture-Process"]
if is_mixture:
    mixture_cols = st.multiselect("Select Mixture Components (sum to 100%)", [f["name"] for f in factor_data], default=[f["name"] for f in factor_data[:3]])
    process_cols = [f["name"] for f in factor_data if f["name"] not in mixture_cols]
else:
    mixture_cols = []
    process_cols = []

# --- STEP 3: Runs ---
fixed_run_designs = ["Full Factorial", "Plackett-Burman"]
if selected_doe in fixed_run_designs:
    runs_input = st.number_input("Number of Runs (auto-set)", value=12, disabled=True)
    st.info("Runs auto-calculated.")
else:
    runs_input = st.number_input("Number of Runs", min_value=6, max_value=200, value=12, step=1)

# --- Generate DOE ---
if st.button("Generate DOE Plan", type="primary"):
    with st.spinner("Generating..."):
        n = len(factor_data)
        actual_runs = runs_input

        if selected_doe == "Latin Hypercube":
            doe = lhs(n, samples=actual_runs)
        elif selected_doe == "Full Factorial":
            doe = ff2n(n)
            actual_runs = len(doe)
        elif selected_doe == "Fractional Factorial":
            base = ff2n(n)
            doe = np.tile(base, (int(np.ceil(actual_runs / len(base))), 1))[:actual_runs] if actual_runs >= len(base) else base[:actual_runs]
        elif selected_doe == "Plackett-Burman":
            doe = pbdesign(n)
            actual_runs = len(doe)
        elif selected_doe in ["Mixture Design", "Combined Mixture-Process"]:
            if len(mixture_cols) < 2:
                st.error("At least 2 mixture components needed.")
                st.stop()
            total_factors = len(mixture_cols) + len(process_cols)
            doe = lhs(total_factors, samples=actual_runs)
            actual_runs = len(doe)

        # Scaling
        scaled = []
        for i, f in enumerate(factor_data):
            if f["type"] == "fixed":
                scaled.append(np.full(actual_runs, f["value"]))
            else:
                col = doe[:, i] if doe.ndim > 1 else doe
                if np.min(col) < 0:
                    col = (col + 1) / 2
                scaled.append(col * (f["max"] - f["min"]) + f["min"])

        df = pd.DataFrame(np.column_stack(scaled).round(3), columns=[f["name"] for f in factor_data])

        if is_mixture and mixture_cols:
            df[mixture_cols] = df[mixture_cols].div(df[mixture_cols].sum(axis=1), axis=0) * 100
            df[mixture_cols] = df[mixture_cols].round(2)

        st.session_state.doe_plan = df
        st.success(f"{selected_doe}: {len(df)} runs")

# --- RESULTS ---
if 'doe_plan' in st.session_state:
    df = st.session_state.doe_plan
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Plan", df.to_csv(index=False).encode(), "doe_plan.csv", "text/csv")

    if len(df.columns) >= 2:
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X Axis", df.columns, index=0, key="2d_x")
        y_col = col2.selectbox("Y Axis", df.columns, index=1, key="2d_y")
        fig_2d = px.scatter(df, x=x_col, y=y_col, title="2D Design Space")
        st.plotly_chart(fig_2d, use_container_width=True)

    if len(df.columns) >= 3:
        with st.expander("🔍 3D Design Space", expanded=False):
            plot_cols = df.columns[:3]
            norm_df = (df[plot_cols] - df[plot_cols].min()) / (df[plot_cols].max() - df[plot_cols].min() + 1e-8)
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=norm_df.iloc[:,0], y=norm_df.iloc[:,1], z=norm_df.iloc[:,2], mode='markers', marker=dict(size=10, color='steelblue')))
            cube_edges = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,0], None, [0,0,1], [1,0,1], [1,1,1], [0,1,1], [0,0,1], None, [0,0,0], [0,0,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1], [0,1,0], [0,1,1]]
            x_e, y_e, z_e = [], [], []
            for point in cube_edges:
                if point is None:
                    x_e.append(None); y_e.append(None); z_e.append(None)
                else:
                    x_e.append(point[0]); y_e.append(point[1]); z_e.append(point[2])
            fig.add_trace(go.Scatter3d(x=x_e, y=y_e, z=z_e, mode='lines', line=dict(color='gray', width=4)))
            fig.update_layout(
                scene=dict(
                    xaxis_title=plot_cols[0],
                    yaxis_title=plot_cols[1],
                    zaxis_title=plot_cols[2],
                    aspectmode='cube',
                    xaxis=dict(range=[-0.05, 1.05]),
                    yaxis=dict(range=[-0.05, 1.05]),
                    zaxis=dict(range=[-0.05, 1.05])
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

# --- ANALYSIS ---
st.divider()
st.subheader("Upload Results & Analyze")

st.info("**Tip:** Response columns should be numeric and have unique names. The app will check for issues.")

if st.button("Enable Analysis Mode", type="secondary"):
    st.session_state.analysis = True

if st.session_state.get("analysis", False):
    uploaded = st.file_uploader("Upload Results CSV", type="csv")
    if uploaded and 'doe_plan' in st.session_state:
        try:
            results = pd.read_csv(uploaded)
            doe_plan = st.session_state.doe_plan
            
            merged = pd.merge(doe_plan.round(6), results.round(6), on=list(doe_plan.columns), how='inner')
            if merged.empty:
                st.error("No matching runs — check column names/values.")
            else:
                st.success(f"Loaded {len(merged)} runs!")
                st.dataframe(merged, use_container_width=True)
                
                response_options = [col for col in merged.columns if col not in doe_plan.columns]
                if not response_options:
                    st.error("No response columns found.")
                    st.stop()
                
                response_col = st.selectbox("Select Response", response_options, key="response_select")

                if st.button("Run Analysis", key="run_analysis"):
                    with st.spinner("Fitting model..."):
                        # Sanitization
                        def sanitize_name(name):
                            name = re.sub(r'\W|^(?=\d)', '_', str(name).strip())
                            if name == '' or name[0].isdigit() or not name[0].isalpha():
                                name = "F_" + name
                            keywords = {'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield', 'True', 'False', 'None'}
                            if name in keywords:
                                name = "Var_" + name
                            return name

                        safe_factor_names = [sanitize_name(name) for name in doe_plan.columns]
                        safe_response = sanitize_name(response_col)

                        # Safe copy — only the selected response
                        merged_safe = merged[doe_plan.columns.tolist() + [response_col]].copy()
                        merged_safe.columns = safe_factor_names + [safe_response]

                        # Ensure response is numeric
                        try:
                            merged_safe[safe_response] = pd.to_numeric(merged_safe[safe_response], errors='coerce')
                            if merged_safe[safe_response].isna().any():
                                st.error("Response column contains non-numeric values (text, blanks, etc.). Fix your CSV.")
                                st.stop()
                        except:
                            st.error("Response column is not numeric.")
                            st.stop()

                        # Formula
                        if is_mixture:
                            terms = ' + '.join(safe_factor_names)
                            inter_terms = ' + '.join([f"{a}:{b}" for i, a in enumerate(safe_factor_names) for b in safe_factor_names[i+1:]])
                            formula = f"{safe_response} ~ {terms} + {inter_terms} - 1"
                        else:
                            formula = f"{safe_response} ~ ({' + '.join(safe_factor_names)})**2"

                        model = ols(formula, data=merged_safe).fit()

                        # Results
                        st.subheader("Model Summary")
                        st.text(model.summary())

                        st.subheader("ANOVA Table")
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        st.dataframe(anova_table.round(4))

                        st.subheader("Coefficients")
                        coeffs = pd.DataFrame({
                            "Term": model.params.index,
                            "Coefficient": model.params.values,
                            "p-value": model.pvalues.values
                        }).round(4)
                        st.dataframe(coeffs)

                        st.write(f"**R² = {model.rsquared:.3f} | Adjusted R² = {model.rsquared_adj:.3f}**")

                        # Contour Plot
                        st.subheader("Response Surface Contour")
                        col1, col2 = st.columns(2)
                        x_disp = col1.selectbox("X Axis", doe_plan.columns, index=0, key="contour_x")
                        y_disp = col2.selectbox("Y Axis", doe_plan.columns, index=1, key="contour_y")

                        x_safe = safe_factor_names[doe_plan.columns.tolist().index(x_disp)]
                        y_safe = safe_factor_names[doe_plan.columns.tolist().index(y_disp)]

                        x_range = np.linspace(merged[x_disp].min(), merged[x_disp].max(), 30)
                        y_range = np.linspace(merged[y_disp].min(), merged[y_disp].max(), 30)
                        X_grid, Y_grid = np.meshgrid(x_range, y_range)

                        grid = pd.DataFrame({x_safe: X_grid.ravel(), y_safe: Y_grid.ravel()})
                        for s_name in safe_factor_names:
                            if s_name not in grid.columns:
                                grid[s_name] = merged_safe[s_name].mean()

                        Z = model.predict(grid).values.reshape(30, 30)

                        fig = go.Figure(data=go.Contour(x=x_range, y=y_range, z=Z, colorscale='Viridis'))
                        fig.update_layout(title=f"Predicted {response_col}", xaxis_title=x_disp, yaxis_title=y_disp)
                        st.plotly_chart(fig, use_container_width=True)

                        # 3D Surface
                        fig_surface = go.Figure(data=go.Surface(x=x_range, y=y_range, z=Z, colorscale='Viridis'))
                        fig_surface.update_layout(title=f"{response_col} 3D Surface", scene=dict(xaxis_title=x_disp, yaxis_title=y_disp, zaxis_title=response_col))
                        st.plotly_chart(fig_surface, use_container_width=True)

                        # Optimum
                        best_idx = np.argmax(Z)
                        best_x, best_y = X_grid.ravel()[best_idx], Y_grid.ravel()[best_idx]
                        st.success(f"**Predicted Optimum:** {x_disp} = {best_x:.2f}, {y_disp} = {best_y:.2f} → {response_col} = {Z.max():.2f}")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Check that response is numeric and column names are unique. Try regenerating the plan if issues persist.")

st.caption("Pocket DOE — Built for real bench work. Made with ❤️ by a former chemist.")
# In[ ]:




