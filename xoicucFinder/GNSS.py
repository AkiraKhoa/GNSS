import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import argparse
import h5py  # Thêm h5py để làm việc với HDF5
from simulation_advanced import generate_advanced_synthetic_gnss_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='GNSS Displacement Detection')
    
    # Thêm tham số đầu vào cho dữ liệu GNSS
    parser.add_argument('--data', type=str, help='Path to the GNSS data file (HDF5 format)')
    
    # Thêm tùy chọn cho mô phỏng dữ liệu GNSS giả
    parser.add_argument('--simulate', action='store_true', help='Generate simulated GNSS data instead of loading from file')
    
    # Thêm tham số đầu vào cho số lượng mẫu MCMC
    parser.add_argument('--samples', type=int, default=2000, help='Number of MCMC samples')
    
    # Thêm tham số đầu vào cho ngưỡng cảnh báo
    parser.add_argument('--alert_threshold', type=float, default=0.05, help='Threshold for displacement alert')
    
    # Thêm tham số đầu vào cho tùy chọn đầu ra
    parser.add_argument('--output_file', type=str, default='output.png', help='Path to the output file for plots')
    
    args = parser.parse_args()
    return args

def load_gnss_data(file_path):
    # Tải dữ liệu từ tệp HDF5
    with h5py.File(file_path, 'r') as hf:
        data = hf['coordinates'][:]
    return data

def main():
    args = parse_arguments()
    
    if args.simulate:
        # Tạo dữ liệu GNSS giả với mô phỏng nâng cao
        data = generate_advanced_synthetic_gnss_data(
            n_points=1000, 
            change_points=[200, 500, 800], 
            mu_values=[[0, 0, 0], [2, -2, 2], [3, 1, -3], [0.5, -0.5, 1]],
            sigma_values=[0.5, 1.0, 0.8, 0.6]
        )
    else:
        # Tải dữ liệu GNSS từ file HDF5
        if not args.data:
            raise ValueError("Please provide a data file with --data or enable simulation with --simulate.")
        data = load_gnss_data(args.data)
    
    # Gọi các hàm tiếp theo với các tham số từ args
    run_bayesian_inference(data, args.samples, args.alert_threshold, args.output_file)

def run_bayesian_inference(data, samples, alert_threshold, output_file):
    # Chúng ta sẽ xử lý từng thành phần (X, Y, Z) riêng biệt
    for i, coord in enumerate(['X', 'Y', 'Z']):
        coord_data = data[:, i]  # Lấy dữ liệu của từng tọa độ
        
        n = len(coord_data)
        
        # Xây dựng mô hình Bayesian
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
            
            # Thực hiện MCMC
            trace = pm.sample(samples, return_inferencedata=True)
        
        # Phân tích và lưu kết quả cho từng tọa độ
        az.plot_trace(trace)
        plt.savefig(f"{output_file}_{coord}.png")  # Lưu kết quả cho từng tọa độ (X, Y, Z)
        print(f"Output for {coord} saved to {output_file}_{coord}.png")

if __name__ == '__main__':
    main()
