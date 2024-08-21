import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import argparse
import h5py
from simulation_advanced import generate_advanced_synthetic_gnss_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='GNSS Displacement Detection')
    parser.add_argument('--data', type=str, help='Path to the GNSS data file (HDF5 format)')
    parser.add_argument('--simulate', action='store_true', help='Generate simulated GNSS data instead of loading from file')
    parser.add_argument('--samples', type=int, default=2000, help='Number of MCMC samples')
    parser.add_argument('--alert_threshold', type=float, default=0.05, help='Threshold for displacement alert')
    parser.add_argument('--output_file', type=str, default='output.png', help='Path to the output file for plots')
    args = parser.parse_args()
    return args

def load_gnss_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        data = hf['coordinates'][:]
    return data


def plot_with_change_points(trace, coord, output_file):
    # Thiết lập kích thước cho toàn bộ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sử dụng tight_layout() để tự động điều chỉnh khoảng cách giữa các subplot
    az.plot_trace(trace, var_names=['mu1', 'mu2', 'mu3', 'tau1', 'tau2'], compact=True)
    
    # Tính toán giá trị trung bình của tau1 và tau2
    posterior_mean_tau1 = np.mean(trace.posterior['tau1'].values)
    posterior_mean_tau2 = np.mean(trace.posterior['tau2'].values)
    
    # Thêm đường thẳng đứng để hiển thị vị trí tau1 và tau2
    plt.axvline(posterior_mean_tau1, color='r', linestyle='--', label=f'Mean tau1: {int(posterior_mean_tau1)}')
    plt.axvline(posterior_mean_tau2, color='g', linestyle='--', label=f'Mean tau2: {int(posterior_mean_tau2)}')
    
    # Hiển thị nhãn chỉ dẫn rõ ràng
    plt.legend(loc='upper right', fontsize=10)  # Di chuyển nhãn để tránh chồng lấn
    
    # Tiêu đề chính cho biểu đồ
    plt.suptitle(f'Trace plot with change points for {coord}', fontsize=16)
    
    # Áp dụng layout để tránh chồng lấn nội dung
    plt.tight_layout()
    
    # Lưu biểu đồ vào file
    plt.savefig(output_file)
    plt.show()

def run_bayesian_inference(data, samples, alert_threshold, output_file):
    for i, coord in enumerate(['X', 'Y', 'Z']):
        coord_data = data[:, i]
        n = len(coord_data)
        
        with pm.Model() as model:
            tau1 = pm.DiscreteUniform('tau1', lower=0, upper=n)
            tau2 = pm.DiscreteUniform('tau2', lower=tau1 + 300, upper=n)
            
            mu1 = pm.Normal('mu1', mu=0, sigma=1)
            mu2 = pm.Normal('mu2', mu=0, sigma=1)
            mu3 = pm.Normal('mu3', mu=0, sigma=1)
            
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            idx = np.arange(n)
            mu = pm.math.switch(tau1 >= idx, mu1, 
                                pm.math.switch(tau2 >= idx, mu2, mu3))
            
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=coord_data)
            
            trace = pm.sample(samples, tune=2000, return_inferencedata=True)
        
        posterior_mean_tau1 = np.mean(trace.posterior['tau1'].values)
        posterior_mean_tau2 = np.mean(trace.posterior['tau2'].values)
        credible_interval_tau1 = az.hdi(trace.posterior['tau1'].values, hdi_prob=0.95)
        credible_interval_tau2 = az.hdi(trace.posterior['tau2'].values, hdi_prob=0.95)
        
        print(f"{coord} - Posterior mean of tau1: {posterior_mean_tau1}")
        print(f"{coord} - 95% Credible interval for tau1: {credible_interval_tau1}")
        print(f"{coord} - Posterior mean of tau2: {posterior_mean_tau2}")
        print(f"{coord} - 95% Credible interval for tau2: {credible_interval_tau2}")
        
        plot_with_change_points(trace, coord, f"{output_file}_{coord}.png")

def main():
    args = parse_arguments()
    
    if args.simulate:
        data = generate_advanced_synthetic_gnss_data(
            n_points=1000, 
            change_points=[200, 500, 800], 
            mu_values=[[0, 0, 0], [2, -2, 2], [3, 1, -3], [0.5, -0.5, 1]],
            sigma_values=[0.5, 1.0, 0.8, 0.6]
        )
    else:
        if not args.data:
            raise ValueError("Please provide a data file with --data or enable simulation with --simulate.")
        data = load_gnss_data(args.data)
    
    run_bayesian_inference(data, args.samples, args.alert_threshold, args.output_file)

if __name__ == '__main__':
    main()
    
# python GNSS.py --simulate --samples 10000 --alert_threshold 0.1 --output_file result.png
