import numpy as np

# Đọc dữ liệu từ tệp numpy với allow_pickle=True
data = np.load('gnss_data.npy', allow_pickle=True)

# Hiển thị thông tin cơ bản về dữ liệu
print("Dữ liệu đã tải:")
print(data)  # Hiển thị toàn bộ dữ liệu (có thể rất dài, bạn có thể xem phần đầu của dữ liệu)
print("Kích thước dữ liệu:", data.shape)  # Hiển thị kích thước của dữ liệu
print("Dữ liệu đầu tiên:", data[:10])  # Hiển thị 10 giá trị đầu tiên để kiểm tra
