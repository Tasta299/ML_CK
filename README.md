- File sku_data.xlsx : file dữ liệu thô
- file sku_data_preprocessed.xlsx : file dữ liệu đã xử lý
- file Tien_xu_ly.ipynb : file code tiền xử lý
  ## Bước 10: Tạo Các Bản Dữ Liệu Bổ Sung Phục Vụ Nghiên Cứu & Đối Soát
Để đáp ứng trọn vẹn yêu cầu kiểm chứng thực nghiệm, chúng ta xuất ra các phiên bản dữ liệu sau:
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


