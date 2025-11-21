"""
Export of `doe app fun.ipynb` cell to standalone Python script.
Run with: streamlit run doe_app_fun.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import lhs

st.set_page_config(page_title="DOE-in-a-Box")
st.title("DOE-in-a-Box")

factors = st.slider("Factors", 2, 6, 3)
runs = st.number_input("Runs", min_value=6, max_value=200, value=12, step=1)
seed = st.number_input("Random seed (0 = none)", min_value=0, value=0, step=1)

if st.button("Generate Design"):
    if seed != 0:
        np.random.seed(int(seed))
    doe = lhs(n=factors, samples=int(runs))
    df = pd.DataFrame(doe.round(3), columns=[f"F{i+1}" for i in range(factors)])
    st.dataframe(df)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="doe.csv", mime="text/csv")
