
  ## HƯỚNG DẪN SỬ DỤNG FILE
1. **Bản 1: `sku_data_preprocessed.xlsx` (Bản đầy đủ chuẩn hóa)**:
   - Các biến liên tục được RobustScaler, `Expire_category` giữ nguyên (0–3), kèm cờ `Is_ghost_sku` và `Is_operational_outlier`.
2. **Bản 2: `sku_data_unscaled_clean.xlsx` (Bản chưa scale)**:
   - Dữ liệu đã làm sạch và Log1p nhưng chưa co giãn qua scaler, phục vụ trực tiếp việc diễn giải ý nghĩa số học và kinh tế thực tế của từng cụm SKU.
3. **Bản 3: `sku_data_active_only.xlsx` (Bản chỉ gồm SKU hoạt động)**:
   - Loại trừ hoàn toàn nhóm Ghost SKU (`Is_ghost_sku == 0`), giúp thuật toán DBSCAN không bị chi phối bởi cụm điểm 0 khổng lồ.
4. **Bản 4: `sku_data_non_multicollinear.xlsx` (Bản loại bỏ đa cộng tuyến)**:
   - Lược bỏ biến trùng lặp `Total outbound_scaled` (giữ lại `Outbound number_scaled` và `Avg_units_per_order_scaled`).
5. **Bản 5: `sku_data_pca_reduced.xlsx` (Bản giảm chiều PCA 90%–95% phương sai)**:
   - Không gian đặc trưng nén giữ lại trên 90%–95% phương sai gốc.
6. **Tien_xu_ly.ipynb** : file code tiền xử lý
7. **test.ipynb** : file code chạy thử

# 2 file mới nhất ( xlsx + ipynb ) là demo

   

## Incremental DBSCAN thủ công

- Dữ liệu đầu vào hoàn chỉnh: `sku_clean_scaled.xlsx` (2.073 SKU đang hoạt động, 16 cột).
- Thuật toán và các chỉ số đánh giá được cài đặt trong `incremental_dbscan.py`. Không dùng DBSCAN, KFold hoặc metric của scikit-learn; NumPy chỉ dùng để tính toán mảng và khoảng cách.
- Mở `Incremental_DBSCAN.ipynb` bằng Jupyter và chạy các ô từ trên xuống. Notebook đọc dữ liệu theo `Stream_order`, dùng 70% đầu để chọn tham số bằng vòng lặp thủ công và cập nhật mô hình bằng ba batch còn lại.
- Cần Python 3 cùng `numpy`, `pandas`, `openpyxl`, `matplotlib`, `jupyter` (hoặc môi trường Jupyter tương đương).
- Quy tắc chọn trong notebook cho kết quả `best_estimator` với `eps=0.65`, `min_samples=12`. Sau ba lần cập nhật: 16 cụm, 445 điểm nhiễu (21,47%), silhouette 0,3455, Davies–Bouldin 0,7300 và Calinski–Harabasz 408,1489. Các chỉ số hình học loại trừ điểm nhiễu.
- Notebook vẽ các cụm cuối cùng trên hai mặt phẳng PCA (PC1–PC2 và PC1–PC3), tô màu từng cụm và hiển thị nhiễu màu xám; hình được lưu thành `clusters_pca.png`. PCA chỉ để trực quan hóa; thuật toán vẫn chạy trên đủ 10 chiều đặc trưng.
- `partial_fit` nhận dữ liệu **đã xử lý theo cùng quy trình** với workbook đầu vào. Repo chưa có bộ scaler gốc để biến đổi SKU mới từ dữ liệu thô; phần minh họa cập nhật dùng 30% dữ liệu giữ lại từ workbook. Thuật toán hỗ trợ thêm điểm, không hỗ trợ xóa điểm hoặc đổi tham số mà không fit lại.


