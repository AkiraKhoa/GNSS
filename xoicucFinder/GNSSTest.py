import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

if __name__ == '__main__':
    # Giả lập dữ liệu chuỗi thời gian với các điểm thay đổi
    np.random.seed(42)
    n = 1000
    tau_true = [300, 600]
    mu_true = [0, 1, -1]
    sigma_true = 0.5

    # Tạo dữ liệu giả lập
    data = np.concatenate([
        np.random.normal(mu_true[0], sigma_true, tau_true[0]),
        np.random.normal(mu_true[1], sigma_true, tau_true[1] - tau_true[0]),
        np.random.normal(mu_true[2], sigma_true, n - tau_true[1])
    ])

    # Xây dựng mô hình Bayesian với PyMC
    with pm.Model() as model:
        # Priors cho các điểm thay đổi
        tau1 = pm.DiscreteUniform('tau1', lower=0, upper=n)
        tau2 = pm.DiscreteUniform('tau2', lower=0, upper=n)
        
        # Priors cho các giá trị mean
        mu1 = pm.Normal('mu1', mu=0, sigma=1)
        mu2 = pm.Normal('mu2', mu=0, sigma=1)
        mu3 = pm.Normal('mu3', mu=0, sigma=1)
        
        # Prior cho standard deviation
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood (xác suất) cho dữ liệu
        idx = np.arange(n)
        mu = pm.math.switch(tau1 >= idx, mu1, 
                            pm.math.switch(tau2 >= idx, mu2, mu3))
        
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
        
        # Thực hiện MCMC
        trace = pm.sample(2000, return_inferencedata=True)

    # Kết quả
    az.plot_trace(trace)
    plt.show()