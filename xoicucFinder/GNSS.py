import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='GNSS Displacement Detection')
    
    # Thêm tham số đầu vào cho dữ liệu GNSS
    parser.add_argument('--data', type=str, required=True, help='Path to the GNSS data file (numpy format)')
    
    # Thêm tham số đầu vào cho số lượng mẫu MCMC
    parser.add_argument('--samples', type=int, default=2000, help='Number of MCMC samples')
    
    # Thêm tham số đầu vào cho ngưỡng cảnh báo
    parser.add_argument('--alert_threshold', type=float, default=0.05, help='Threshold for displacement alert')
    
    # Thêm tham số đầu vào cho tùy chọn đầu ra
    parser.add_argument('--output_file', type=str, default='output.png', help='Path to the output file for plots')
    
    args = parser.parse_args()
    return args

def load_gnss_data(file_path):
    # Giả sử dữ liệu GNSS được lưu trong tệp numpy
    return np.load(file_path, allow_pickle=True)

def main():
    args = parse_arguments()
    
    # Tải dữ liệu GNSS
    data = load_gnss_data(args.data)
    
    # Gọi các hàm tiếp theo với các tham số từ args
    run_bayesian_inference(data, args.samples, args.alert_threshold, args.output_file)

def run_bayesian_inference(data, samples, alert_threshold, output_file):
    n = len(data)
    
    # Xây dựng mô hình Bayesian
    with pm.Model() as model:
        tau1 = pm.DiscreteUniform('tau1', lower=0, upper=n)
        tau2 = pm.DiscreteUniform('tau2', lower=0, upper=n)
        
        mu1 = pm.Normal('mu1', mu=0, sigma=1)
        mu2 = pm.Normal('mu2', mu=0, sigma=1)
        mu3 = pm.Normal('mu3', mu=0, sigma=1)
        
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        idx = np.arange(n)
        mu = pm.math.switch(tau1 >= idx, mu1, 
                            pm.math.switch(tau2 >= idx, mu2, mu3))
        
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
        
        # Thực hiện MCMC
        trace = pm.sample(samples, return_inferencedata=True)
    
    # Phân tích và lưu kết quả
    az.plot_trace(trace)
    plt.savefig(output_file)
    print(f"Output saved to {output_file}")

if __name__ == '__main__':
    main()
