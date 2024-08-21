import numpy as np

def generate_advanced_synthetic_gnss_data(n_points=1000, change_points=None, mu_values=None, sigma_values=None):
    """
    Tạo dữ liệu GNSS giả lập nâng cao với các điểm thay đổi và nhiễu khác nhau.
    :param n_points: Số lượng điểm dữ liệu
    :param change_points: Danh sách các điểm thay đổi
    :param mu_values: Danh sách các giá trị trung bình (mean) cho mỗi đoạn dữ liệu
    :param sigma_values: Danh sách các độ lệch chuẩn (sigma) cho mỗi đoạn dữ liệu
    :return: Mảng dữ liệu GNSS (X, Y, Z)
    """
    if change_points is None:
        change_points = [300, 600]
    
    if mu_values is None:
        mu_values = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
    
    if sigma_values is None:
        sigma_values = [0.5, 0.7, 1.0]

    data = np.zeros((n_points, 3))  # Dữ liệu cho X, Y, Z
    
    for i in range(3):  # Tạo dữ liệu cho từng tọa độ X, Y, Z
        start = 0
        for j, change_point in enumerate(change_points):
            end = change_point
            data[start:end, i] = np.random.normal(mu_values[j][i], sigma_values[j], end - start)
            start = end
        # Phần cuối cùng của dữ liệu sau điểm thay đổi cuối cùng
        data[start:, i] = np.random.normal(mu_values[-1][i], sigma_values[-1], n_points - start)
    
    return data
