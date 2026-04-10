import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regressionfunctions import (
    em_gmm, em_gmm_noise, em_gmm_huber, em_gmm_consensus, calculate_best_method
)

st.set_page_config(page_title="Robust Regression Lab", layout="wide")

st.title("Custom Robust Mixture Regression")
st.markdown("""
Add your own data points in the table below. The app will automatically test four different 
robust EM algorithms and pick the one that best fits your data.
""")

with st.sidebar:
    st.header("Algorithm Settings")
    k_comp = st.slider("Number of Regression Lines (k)", 1, 3, 2)
    max_iter = st.number_input("Max Iterations", 10, 500, 100)
    
    st.divider()
    st.header("Parameters")
    huber_delta = st.slider("Huber Delta", 0.1, 3.0, 1.4)
    cons_width = st.slider("Consensus Tube Width", 0.1, 5.0, 1.0)

st.subheader("1. Input Your Data")

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
"x": [7.61, 2.50, 0.25, 3.05, 4.94, 4.09, 3.60, 9.55, 2.91, 6.08], # ... + 210 more
        "y": [16.09, 7.61, 14.45, 7.12, 10.47, 2.85, 4.13, -13.49, 6.09, 14.09]    })

edited_df = st.data_editor(
    st.session_state.df, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "x": st.column_config.NumberColumn(help="Input Feature"),
        "y": st.column_config.NumberColumn(help="Target Value")
    }
)

if st.button("Run Regression Analysis"):
    X_raw = edited_df["x"].values.reshape(-1, 1)
    y = edited_df["y"].values
    X = np.hstack((np.ones((len(y), 1)), X_raw))
    
    with st.spinner("Analyzing data with multiple algorithms..."):
        res_trim = em_gmm(X, y, k_comp, 0.1, max_iter)
        res_noise = em_gmm_noise(X, y, k_comp, max_iter)
        res_huber = em_gmm_huber(X, y, k_comp, huber_delta, max_iter)
        res_cons = em_gmm_consensus(X, y, k_comp, max_iter, cons_width)

        results = {
            "Trimmed EM": res_trim,
            "Noise Component EM": res_noise,
            "Huber Weighted EM": res_huber,
            "Consensus EM": res_cons
        }

        best_name = ""
        min_mae = float('inf')
        
        for name, (betas, sigmas, pis) in results.items():
            mae = calculate_best_method(X, y, betas, k_comp)
            if mae < min_mae:
                min_mae = mae
                best_name = name
                best_betas, best_sigmas, best_pis = betas, sigmas, pis

    st.success(f"**Winner:** {best_name} (MAE: {min_mae:.4f})")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(edited_df["x"], edited_df["y"], color='darkblue', alpha=0.6, label='User Data')
    
    x_plot = np.linspace(edited_df["x"].min(), edited_df["x"].max(), 100)
    for j in range(k_comp):
        y_plot = best_betas[j][0] + best_betas[j][1] * x_plot
        ax.plot(x_plot, y_plot, linewidth=3, label=f"Line {j+1} (Weight: {best_pis[j]:.2f})")
    
    ax.set_title(f"Best Fit: {best_name}")
    ax.legend()
    st.pyplot(fig)

    cols = st.columns(k_comp)
    for i in range(k_comp):
        cols[i].metric(f"Line {i+1} Slope", f"{best_betas[i][1]:.3f}")
        cols[i].write(f"Sigma (Noise): {best_sigmas[i]:.3f}")
