# GNSS Displacement Detection

## Mô Tả

Dự án này sử dụng PyMC3 để phát hiện sự thay đổi trong chuỗi thời gian dữ liệu GNSS. Dữ liệu được giả lập và mô hình Bayesian được xây dựng để phân tích dữ liệu.

## Cài Đặt Môi Trường

Để cài đặt môi trường và các phụ thuộc cần thiết, làm theo các bước sau:

1. **Cài đặt Conda**: Nếu bạn chưa có Conda, tải và cài đặt Miniconda hoặc Anaconda từ trang web chính thức của chúng.

2. **Tạo và kích hoạt môi trường Conda**:

   ```bash
   conda env create -f environment.yml
   conda activate pymc_env
