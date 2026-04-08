import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor


##st.title("Interactive Robust Mixture Regression")
##st.markdown("""
##This app demonstrates Expectation-Maximization (EM) for mixture regressions 
##with an added trimming step for  robustness.
##""")
##
##with st.sidebar:
##    st.header("1. Data Generation")
##    n_samples = st.slider("Total Sample Size", 200, 5000, 1000)
##    k_components = st.slider("Number of Components (k)", 1, 4, 2)
##    outlier_prop = st.slider("Outlier Proportion", 0.0, 0.20, 0.05, step=0.01)
##    noise_level = st.slider("Data Noise (Sigma)", 0.1, 2.0, 0.5)
##
##    st.header("2. Algorithm Settings")
##    max_iter = st.number_input("Max EM Iterations", 10, 500, 100)
##    seed = st.number_input("Random Seed", 0, 100, 3)
##width = st.sidebar.slider("Consensus Width (Tube)", 0.05, 3.0, 2.0)
##np.random.seed(seed)


def generate_data(n,k,prop,sigma):
    X = np.hstack((np.ones((n, 1)), np.random.rand(n, 1)))
    #true_betas = [np.array([0, -6]), np.array([-5, 10])]
    true_betas = [np.array([np.random.uniform(-10, 10), np.random.uniform(-20, 20)]) for _ in range(k)]
    iss = np.array_split(np.random.permutation(n), k)
    y = np.zeros(n)
    for j in range(k):
        idx = iss[j]
        y[idx] = X[idx] @ true_betas[j] + sigma * np.random.randn(len(idx))    
    n_outliers = int(prop * n)
    if n_outliers > 0:
        ox = np.hstack((np.ones((n_outliers, 1)), np.random.rand(n_outliers, 1)))
        oy = np.random.uniform(y.min() - 5, y.max() + 5, n_outliers)
        X = np.vstack((X, ox))
        y = np.hstack((y, oy))
    else:
        ox, oy = np.empty((0, 2)), np.empty(0)
        
    return X, y, ox, oy, true_betas
def em_gmm(X, y, k, outlier_proportion, iterations):
    n_samples, n_features = X.shape
    
    betalist = [np.random.rand(n_features) for _ in range(k)]
    pilist = np.random.dirichlet(np.ones(k))
    sigmalist = np.random.rand(k)
    
    gammalist = np.zeros((len(y), k))
    
    for p in range(iterations):
        for j in range(k):
            gammalist[:, j] = pilist[j] * norm.pdf(y, X @ betalist[j], sigmalist[j])
        
        row_sums = gammalist.sum(axis=1, keepdims=True)
        gammalist = gammalist / (row_sums + 1e-10) 
        
        all_residuals = np.array([np.abs(y - X @ betalist[j])**2 for j in range(k)])
        min_residuals = np.min(all_residuals, axis=0)
        
        trim_threshold = np.percentile(min_residuals, 100 - 100 * outlier_proportion)
        inliers = min_residuals <= trim_threshold
        
        X_inliers = X[inliers]
        y_inliers = y[inliers]
        
        for j in range(k):
            W_j = gammalist[inliers, j]
            N_j = np.sum(W_j)
            
            if N_j > 1e-5:
                XTWX = (X_inliers.T * W_j) @ X_inliers
                XTWY = (X_inliers.T * W_j) @ y_inliers
                
                betalist[j] = np.linalg.pinv(XTWX) @ XTWY
                
                variance = np.sum(W_j * (y_inliers - X_inliers @ betalist[j])**2) / N_j
                sigmalist[j] = np.sqrt(variance)
        
        pilist = [np.sum(gammalist[inliers, j]) / np.sum(inliers) for j in range(k)]

    
    return betalist, sigmalist, pilist








def em_gmm_huber(X, y, k, delta, iterations):
    n_samples, n_features = X.shape
    
    betalist = [np.random.rand(n_features) for _ in range(k)]
    pilist = np.random.dirichlet(np.ones(k))
    sigmalist = np.random.rand(k)
    gammalist = np.zeros((len(y), k))
    
    for p in range(iterations):
        for j in range(k):
            gammalist[:, j] = pilist[j] * norm.pdf(y, X @ betalist[j], sigmalist[j])
        
        row_sums = gammalist.sum(axis=1, keepdims=True)
        gammalist = gammalist / (row_sums + 1e-10) 
        
##        all_residuals = np.array([np.abs(y - X @ betalist[j])**2 for j in range(k)])
##        min_residuals = np.min(all_residuals, axis=0)
##        
##        trim_threshold = np.percentile(min_residuals, 100 - 100 * outlier_proportion)
##        inliers = min_residuals <= trim_threshold
##        
##        X_inliers = X[inliers]
##        y_inliers = y[inliers]
##        
        for j in range(k):
            residuals=np.abs(y-X @ betalist[j])
            scaled_delta=delta*sigmalist[j]
            weights=np.where(residuals <= scaled_delta,1.0,scaled_delta/(residuals+1e-10))
            W_j=gammalist[:,j] * weights
            N_j=np.sum(W_j)
            
            if (N_j > 1e-5):
                XTWX = (X.T * W_j) @ X
                XTWY = (X.T * W_j) @ y
                
                betalist[j] = np.linalg.pinv(XTWX) @ XTWY
                
                variance = np.sum(W_j * (y - X @ betalist[j])**2) / N_j
                sigmalist[j] = np.sqrt(variance)
        
        pilist = gammalist.mean(axis=0)
    
    return betalist, sigmalist, pilist



def em_gmm_noise(X, y, k, iterations):
    n_samples, n_features = X.shape
    pilist = np.ones(k + 1) / (k + 1) 
    betalist = [np.random.rand(n_features) for _ in range(k)]
    sigmalist = np.random.rand(k)
    
    y_range = y.max() - y.min()
    noise_density = 1.0 / y_range

    for p in range(iterations):
        gammalist = np.zeros((n_samples, k + 1))
        for j in range(k):
            gammalist[:, j] = pilist[j] * norm.pdf(y, X @ betalist[j], sigmalist[j])
        
        gammalist[:, k] = pilist[k] * noise_density
        
        gammalist /= gammalist.sum(axis=1, keepdims=True)

        for j in range(k):
            W_j = gammalist[:, j]
            N_j = np.sum(W_j)
            
            if N_j > 1e-5:
                XTWX = (X.T * W_j) @ X
                XTWY = (X.T * W_j) @ y
                betalist[j] = np.linalg.pinv(XTWX) @ XTWY
                
                variance = np.sum(W_j * (y - X @ betalist[j])**2) / N_j
                sigmalist[j] = np.sqrt(variance)

        pilist = gammalist.mean(axis=0)

    return betalist, sigmalist, pilist


def em_gmm_consensus(X, y, k, iterations, width=0.5):
    n_samples, n_features = X.shape
    
    def run_once():
        betalist = []
        iss = [np.random.choice(n_samples)]
        
        for i in range(k):
            # small local cluster
            dists = np.linalg.norm(X[:, 1:] - X[iss[-1], 1:], axis=1)
            local_idx = np.argsort(dists)[:20]
            X_sub, y_sub = X[local_idx], y[local_idx]
            beta = (np.linalg.pinv(X_sub.T @ X_sub + 1e-2*np.eye(n_features)) @ X_sub.T) @ y_sub
            betalist.append(beta)
            #new id
            iss.append(np.argmax(dists))

        for i in range(iterations):
            residuals = np.array([np.abs(y - X @ b) for b in betalist])
            
            assignments = np.argmin(residuals, axis=0)
            min_residuals = np.min(residuals, axis=0)
            
            inlier_mask = min_residuals < width

            for j in range(k):
                mask = (assignments == j) & inlier_mask
                
                if np.sum(mask) > n_features + 5:
                    # weighted fit
                    X_j, y_j = X[mask], y[mask]
                    betalist[j] = np.linalg.pinv(X_j.T @ X_j) @ X_j.T @ y_j
                else:
                    unexplained_idx = np.argmax(min_residuals)
                    d = np.linalg.norm(X[:, 1:] - X[unexplained_idx, 1:], axis=1)
                    new_local = np.argsort(d)[:20]
                    betalist[j] = np.linalg.pinv(X[new_local].T @ X[new_local] + 1e-2*np.eye(n_features)) @ X[new_local].T @ y[new_local]

        # score of run by number of points
        score = np.sum(inlier_mask)
        return betalist, score

    best_betas = None
    best_score = -1
    
    for i in range(20):
        #np.random.seed(i)
        current_betas, current_score = run_once()
        if current_score > best_score:
            best_score = current_score
            best_betas = current_betas

    final_res = np.array([np.abs(y - X @ b) for b in best_betas])
    final_assign = np.argmin(final_res, axis=0)
    sigmalist = [np.std(y[final_assign == j] - X[final_assign == j] @ best_betas[j]) + 1e-3 for j in range(k)]
    pilist = [np.mean(final_assign == j) for j in range(k)]

    return best_betas, sigmalist, pilist

def calculate_best_method(X,y,betalist,k):
    min_residuals = np.min(np.array([np.abs(y-X@betalist[j]) for j in range(k)]), axis=0)
    return np.mean(min_residuals)
    



##X, y, ox, oy, ground_truth = generate_data(n_samples, k_components, outlier_prop, noise_level)
##delta1=1.4
##    
##with st.spinner("Calculating..."):
##    final_betas, final_sigmas, final_pis = em_gmm(X, y, k_components, outlier_prop, max_iter)
##    final_betas_noise, final_sigmas_noise, final_pis_noise = em_gmm_noise(X, y, k_components, max_iter)
##    final_betas_huber, final_sigmas_huber, final_pis_huber = em_gmm_huber(X, y, k_components,delta1, max_iter)
##    final_betas_huber1, final_sigmas_huber1, final_pis_huber1 = em_gmm_consensus(X, y, k_components, max_iter, width)
##
##
##
##mae_trim = calculate_best_method(X,y,final_betas,k_components)
##mae_trim_noise = calculate_best_method(X,y,final_betas_noise,k_components)
##mae_trim_huber = calculate_best_method(X,y,final_betas_huber,k_components)
##mae_trim_huber1 = calculate_best_method(X,y,final_betas_huber1,k_components)
##
##
####fig, ax = plt.subplots(figsize=(10, 5))
####ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
####if len(oy) > 0:
####    ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')
####
####x_plot = np.linspace(0, 1, 100)
####for j in range(k_components):
####    y_plot = final_betas_huber[j][0] + final_betas_huber[j][1] * x_plot
####    ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={final_pis[j]:.2f})")
####ax.set_title(f"EM Mixture Regression (k={k_components}) (huber regression)")
####ax.set_xlabel("X (Feature)")
####ax.set_ylabel("y (Target)")
####ax.legend()
####st.pyplot(fig)
##
##    
##if(mae_trim < mae_trim_noise and mae_trim < mae_trim_huber and mae_trim<mae_trim_huber1):
##
##    fig, ax = plt.subplots(figsize=(10, 5))
##    ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
##    if len(oy) > 0:
##        ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')
##
##    x_plot = np.linspace(0, 1, 100)
##    for j in range(k_components):
##        y_plot = final_betas[j][0] + final_betas[j][1] * x_plot
##        ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={final_pis[j]:.2f})")
##
##    ax.set_title(f"EM Mixture Regression (k={k_components}) (normal em)")
##    ax.set_xlabel("X (Feature)")
##    ax.set_ylabel("y (Target)")
##    ax.legend()
##    st.pyplot(fig)
##elif(mae_trim_noise < mae_trim and mae_trim_noise < mae_trim_huber and mae_trim_noise< mae_trim_huber1):
##
##    fig, ax = plt.subplots(figsize=(10, 5))
##    ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
##    if len(oy) > 0:
##        ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')
##
##    x_plot = np.linspace(0, 1, 100)
##    for j in range(k_components):
##        y_plot = final_betas_noise[j][0] + final_betas_noise[j][1] * x_plot
##        ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={final_pis[j]:.2f})")
##
##    ax.set_title(f"EM Mixture Regression (k={k_components}) (noise regression)")
##    ax.set_xlabel("X (Feature)")
##    ax.set_ylabel("y (Target)")
##    ax.legend()
##    st.pyplot(fig)
##elif (mae_trim_huber<mae_trim_huber1):
##    fig, ax = plt.subplots(figsize=(10, 5))
##    ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
##    if len(oy) > 0:
##        ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')
##
##    x_plot = np.linspace(0, 1, 100)
##    for j in range(k_components):
##        y_plot = final_betas_huber[j][0] + final_betas_huber[j][1] * x_plot
##        ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={final_pis[j]:.2f})")
##
##    ax.set_title(f"EM Mixture Regression (k={k_components}) (huber regression)")
##    ax.set_xlabel("X (Feature)")
##    ax.set_ylabel("y (Target)")
##    ax.legend()
##    st.pyplot(fig)
##else:
##    fig, ax = plt.subplots(figsize=(10, 5))
##    ax.scatter(X[:, 1], y, color='lightgrey', alpha=0.4, label='Data Points')
##    if len(oy) > 0:
##        ax.scatter(ox[:, 1], oy, color='red', s=10, alpha=0.5, label='Outliers')
##
##    x_plot = np.linspace(0, 1, 100)
##    for j in range(k_components):
##        y_plot = final_betas_huber1[j][0] + final_betas_huber1[j][1] * x_plot
##        ax.plot(x_plot, y_plot, linewidth=3, label=f"Comp {j+1} (π={final_pis[j]:.2f})")
##
##    ax.set_title(f"EM Mixture Regression (k={k_components}) (consensus regression)")
##    ax.set_xlabel("X (Feature)")
##    ax.set_ylabel("y (Target)")
##    ax.legend()
##    st.pyplot(fig)
##cols = st.columns(k_components)
##for i in range(k_components):
##    cols[i].metric(f"Component {i+1} Slope", f"{final_betas[i][1]:.3f}")
##    cols[i].write(f"Sigma: {final_sigmas[i]:.3f}")
##
