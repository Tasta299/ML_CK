
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

- Dữ liệu đầu vào: sheet `sku_tien_xu_ly` của `sku_tien_xu_ly_lan1.xlsx` (2.073 SKU).
- Thuật toán và các chỉ số đánh giá được cài đặt trong `incremental_dbscan.py`. 
- Mở **`Incremental_DBSCAN.ipynb`** bằng Jupyter và chạy các ô từ trên xuống. Workbook không có cột thời gian hoặc `Stream_order`, nên notebook tạo thứ tự cập nhật giả lập bằng xáo trộn cố định `seed=42`. 70% đầu được dùng để chọn tham số bằng vòng lặp thủ công; 30% còn lại chia thành ba batch cập nhật.
- Cần Python 3 cùng `numpy`, `pandas`, `openpyxl`, `matplotlib`, `jupyter` (hoặc môi trường Jupyter tương đương).
- Cấu hình chọn được: `eps=0.65`, `min_samples=12`. Sau ba batch: 16 cụm, 445 điểm nhiễu (21,47%), silhouette 0,3455, Davies–Bouldin 0,7300 và Calinski–Harabasz 408,1489. Ba chỉ số hình học loại trừ điểm nhiễu.
- Kết quả xuất từ notebook: `sku_cluster_labels.csv` (nhãn theo ID), `update_results.csv` (chỉ số theo batch), `clusters_pca.png` và `update_summary.png` (biểu đồ). PCA chỉ để trực quan hóa; phân cụm vẫn chạy trên đủ 10 chiều đặc trưng.
- `partial_fit` nhận dữ liệu **đã xử lý theo cùng quy trình** với workbook đầu vào. Chưa có tham số của bộ scaler gốc để biến đổi SKU mới từ dữ liệu thô; phần minh họa cập nhật dùng 30% dữ liệu giữ lại từ workbook. Thuật toán hỗ trợ thêm điểm, không hỗ trợ xóa điểm hoặc đổi tham số mà không fit lại.


