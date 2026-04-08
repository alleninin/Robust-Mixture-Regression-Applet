import streamlit as st
st.set_page_config(
    page_title="Regression Applet",
    page_icon="💀",
)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from regressionfunctions import(
    generate_data, em_gmm, em_gmm_huber, 
    em_gmm_noise, em_gmm_consensus, calculate_best_method
)


#pg=st.navigation([st.Page("customregrerssionmodifiers.py")])
##st.sidebar.success("select which app you prefer")
st.title("Interactive Robust Mixture Regression")
st.markdown("""
This app demonstrates Expectation-Maximization (EM) for mixture regressions 
with an added trimming step for robustness.
""")

with st.sidebar:
    st.header("1. Data Generation")
    n_samples = st.slider("Total Sample Size", 200, 5000, 1000)
    k_components = st.slider("Number of Components (k)", 1, 4, 2)
    outlier_prop = st.slider("Outlier Proportion", 0.0, 0.20, 0.05, step=0.01)
    noise_level = st.slider("Data Noise (Sigma)", 0.1, 2.0, 0.5)

    st.header("2. Algorithm Settings")
    max_iter = st.number_input("Max EM Iterations", 10, 500, 100)
    seed = st.number_input("Random Seed", 0, 100, 3)
    width = st.slider("Consensus Width (Tube)", 0.05, 3.0, 2.0)

np.random.seed(seed)



X, y, ox, oy, ground_truth = generate_data(n_samples, k_components, outlier_prop, noise_level)
delta1=1.4
    
with st.spinner("Calculating..."):
    final_betas, final_sigmas, final_pis = em_gmm(X, y, k_components, outlier_prop, max_iter)
    final_betas_noise, final_sigmas_noise, final_pis_noise = em_gmm_noise(X, y, k_components, max_iter)
    final_betas_huber, final_sigmas_huber, final_pis_huber = em_gmm_huber(X, y, k_components,delta1, max_iter)
    final_betas_huber1, final_sigmas_huber1, final_pis_huber1 = em_gmm_consensus(X, y, k_components, max_iter, width)



mae_trim = calculate_best_method(X,y,final_betas,k_components)
mae_trim_noise = calculate_best_method(X,y,final_betas_noise,k_components)
mae_trim_huber = calculate_best_method(X,y,final_betas_huber,k_components)
mae_trim_huber1 = calculate_best_method(X,y,final_betas_huber1,k_components)


##fig, ax = plt.subplots(figsize=(10, 5))
##ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
##if len(oy) > 0:
##    ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')
##
##x_plot = np.linspace(0, 1, 100)
##for j in range(k_components):
##    y_plot = final_betas_huber[j][0] + final_betas_huber[j][1] * x_plot
##    ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={final_pis[j]:.2f})")
##ax.set_title(f"EM Mixture Regression (k={k_components}) (huber regression)")
##ax.set_xlabel("X (Feature)")
##ax.set_ylabel("y (Target)")
##ax.legend()
##st.pyplot(fig)

def plotplot(betalist, title, pis):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
    if len(oy) > 0:
        ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')

    x_plot = np.linspace(0, 1, 100)
    for j in range(k_components):
        y_plot = betalist[j][0] + betalist[j][1] * x_plot
        ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={pis[j]:.2f})")

    ax.set_title(f"EM Mixture Regression (k={k_components}) ({title})")
    ax.set_xlabel("X (Feature)")
    ax.set_ylabel("y (Target)")
    ax.legend()
    st.pyplot(fig)

    
if(mae_trim < mae_trim_noise and mae_trim < mae_trim_huber and mae_trim<mae_trim_huber1):
    plotplot(final_betas,"normal em", final_pis)
elif(mae_trim_noise < mae_trim and mae_trim_noise < mae_trim_huber and mae_trim_noise< mae_trim_huber1):
    plotplot(final_betas_noise,"noise regression", final_pis_noise)

elif (mae_trim_huber<mae_trim_huber1):
    plotplot(final_betas_huber,"huber regression", final_pis_huber)

else:
    plotplot(final_betas_huber1,"consensus regression", final_pis_huber1)

cols = st.columns(k_components)
for i in range(k_components):
    cols[i].metric(f"Component {i+1} Slope", f"{final_betas[i][1]:.3f}")
    cols[i].write(f"Sigma: {final_sigmas[i]:.3f}")

