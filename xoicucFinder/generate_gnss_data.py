import numpy as np
import h5py

def generate_synthetic_gnss_data(n_points=1000, tau=[300, 600], mu=[0, 1, -1], sigma=0.5):
    """
    Tạo dữ liệu GNSS giả lập với các điểm thay đổi (change points).
    :param n_points: Số lượng điểm dữ liệu
    :param tau: Danh sách các điểm thay đổi
    :param mu: Danh sách các giá trị trung bình (mean) cho mỗi đoạn dữ liệu
    :param sigma: Độ lệch chuẩn của phân phối Gaussian
    :return: Mảng dữ liệu GNSS (X, Y, Z)
    """
    np.random.seed(42)
    
    # Tạo dữ liệu X, Y, Z giả lập với các điểm thay đổi
    x_data = np.concatenate([
        np.random.normal(mu[0], sigma, tau[0]),
        np.random.normal(mu[1], sigma, tau[1] - tau[0]),
        np.random.normal(mu[2], sigma, n_points - tau[1])
    ])
    
    y_data = np.concatenate([
        np.random.normal(mu[0], sigma, tau[0]),
        np.random.normal(mu[1], sigma, tau[1] - tau[0]),
        np.random.normal(mu[2], sigma, n_points - tau[1])
    ])
    
    z_data = np.concatenate([
        np.random.normal(mu[0], sigma, tau[0]),
        np.random.normal(mu[1], sigma, tau[1] - tau[0]),
        np.random.normal(mu[2], sigma, n_points - tau[1])
    ])
    
    # Kết hợp X, Y, Z thành một mảng 2D
    data = np.vstack([x_data, y_data, z_data]).T
    
    return data

def save_data_to_hdf5(data, file_name='synthetic_gnss_data.h5'):
    """
    Lưu dữ liệu vào tệp HDF5.
    :param data: Dữ liệu GNSS cần lưu
    :param file_name: Tên tệp HDF5
    """
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('coordinates', data=data)
    print(f"Data saved to {file_name}")

if __name__ == '__main__':
    # Tạo dữ liệu GNSS giả lập
    synthetic_data = generate_synthetic_gnss_data()
    
    # Lưu dữ liệu vào tệp HDF5
    save_data_to_hdf5(synthetic_data)
