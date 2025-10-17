# **Tài liệu dự án: Pipeline Huấn luyện Phân tán cho các mô hình Deep Learning trên Spark**

*Phiên bản: 2.0 (Kiến trúc Streaming)*
*Công nghệ: Spark 3.5.6, Hadoop 3.4.1, Python 3.10.12*
*Cấu hình cụm tham khảo: 1 Master, 3 Worker (16 Cores, 256GB RAM mỗi node)*

---

## **1. Tổng quan**

Dự án này là một pipeline hoàn chỉnh, mạnh mẽ và có khả năng mở rộng để huấn luyện các mô hình học sâu (tập trung vào các kiến trúc dựa trên RNN) một cách phân tán trên một cụm Apache Spark.

Pipeline được thiết kế theo dạng module hóa, điều khiển bởi cấu hình, và sử dụng kiến trúc nạp dữ liệu kiểu streaming để xử lý các tập dữ liệu lớn hơn bộ nhớ một cách hiệu quả.

### **Các tính năng chính:**

*   **Kiến trúc Module hóa:** Mã nguồn được tổ chức sạch sẽ, phân tách rõ ràng các mối quan tâm (dữ liệu, mô hình, huấn luyện, tiện ích).
*   **Hỗ trợ đa mô hình:** Dễ dàng huấn luyện và so sánh các mô hình `RNN`, `LSTM`, `GRU`, và `Transformer`.
*   **Điều khiển bởi Cấu hình:** Toàn bộ pipeline được điều khiển bởi một tệp `config.yaml` duy nhất, giúp việc thử nghiệm trở nên cực kỳ linh hoạt.
*   **Data Loading hiệu suất cao:** Sử dụng `IterableDataset` để đọc trực tiếp dữ liệu từ các tệp Parquet theo kiểu streaming, loại bỏ hoàn toàn giới hạn về bộ nhớ và thời gian khởi động chậm của phương pháp `InMemory`.
*   **Huấn luyện phân tán ổn định:** Sử dụng `TorchDistributor` kết hợp với cơ chế `steps_per_epoch` và `itertools` để đảm bảo quá trình huấn luyện đồng bộ, tránh hoàn toàn các lỗi deadlock.
*   **Tự động hóa & Báo cáo:** Tự động chuẩn bị dữ liệu, tạo báo cáo, vẽ biểu đồ và lưu lại các mô hình tốt nhất.
*   **Logging chuyên nghiệp:** Tích hợp hệ thống `logging` chi tiết để dễ dàng theo dõi và gỡ lỗi hiệu suất.

## **2. Cấu trúc dự án**

Dự án được tổ chức theo cấu trúc tiêu chuẩn, giúp dễ dàng quản lý và mở rộng:

```
/
├── project/                    # Mã nguồn chính của ứng dụng
│   ├── data/                   # Module xử lý dữ liệu (loader, preprocessor)
│   ├── models/                 # Module chứa các kiến trúc mô hình và "nhà máy" model
│   ├── training/               # Module chứa logic huấn luyện và đánh giá
│   └── utils/                  # Các hàm tiện ích (config, hdfs, logging, visualization)
├── config.yaml                 # Tệp cấu hình trung tâm cho toàn bộ pipeline
├── main.py                     # Trình khởi chạy đơn giản (wrapper)
├── README.md                   # Tài liệu dự án
├── requirements.txt            # Danh sách các gói phụ thuộc Python
└── run.sh                      # Script chính để thực thi pipeline
```

## **3. Cài đặt và Thiết lập**

1.  **Tạo môi trường Python:**
    *   Đảm bảo một môi trường Python (ví dụ: Conda, venv) đã được cài đặt và kích hoạt trên tất cả các node của cụm.
    *   Ví dụ: `python3 -m venv spark_env && source spark_env/bin/activate`

2.  **Cài đặt các gói phụ thuộc:**
    *   Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết từ tệp `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## **4. Hướng dẫn sử dụng và Cấu hình**

### **4.1. Tệp cấu hình `config.yaml`**

Đây là trung tâm điều khiển của pipeline. Bạn có thể thay đổi mọi thứ ở đây mà không cần sửa mã nguồn.

*   **`spark`**: Chứa các cấu hình để khởi chạy công việc Spark. Xem mục 4.2 để biết cấu hình khuyến nghị.
*   **`data`**: Định nghĩa tất cả các đường dẫn đến dữ liệu đầu vào và thư mục đầu ra trên HDFS.
*   **`training`**: Chứa các tham số cho quá trình huấn luyện như `model_type`, `epochs`, `batch_size`, `learning_rate`.
*   **`model_params`**: Cho phép định nghĩa siêu tham số cho từng mô hình. Khối `default` chứa các giá trị mặc định. Các khối riêng cho từng mô hình (`lstm`, `gru`,...) sẽ **ghi đè** lên giá trị mặc định, cho phép thử nghiệm linh hoạt.

### **4.2. Tối ưu hóa Cấu hình Spark**

Để đạt hiệu suất tốt nhất, bạn nên cấu hình tài nguyên Spark một cách hợp lý. Thay vì yêu cầu một số lượng rất lớn các executor yếu, hãy cân nhắc sử dụng **ít executor hơn nhưng mạnh hơn** (nhiều core và memory hơn) để giảm chi phí giao tiếp và đồng bộ hóa.

**Cấu hình được khuyến nghị cho `config.yaml`:**

```yaml
spark:
  master: yarn
  deploy_mode: cluster
  
  # Bật tính năng phân bổ động để Spark tự điều chỉnh tài nguyên
  dynamic_allocation_enabled: true
  shuffle_service_enabled: true
  dynamic_allocation_min_executors: 2
  dynamic_allocation_max_executors: 12 # Điều chỉnh tùy theo cụm
  
  # Cấu hình cho mỗi executor "mập"
  executor_memory: 32G
  executor_cores: 5
  executor_memory_overhead: 4G
  
  driver_memory: 4G
  python_env: /home/ubuntu/spark_env/bin/python # Đường dẫn tới môi trường Python của bạn
```

### **4.3. Chạy Pipeline**

1.  **Chỉnh sửa `config.yaml`**: Mở tệp và điều chỉnh các tham số theo ý muốn.
2.  **Cấp quyền thực thi cho script:** (Chỉ cần làm một lần)
    ```bash
    chmod +x run.sh
    ```
3.  **Chạy pipeline:**
    ```bash
    ./run.sh
    ```

## **5. Luồng hoạt động và Xử lý dữ liệu (Kiến trúc Streaming)**

Đây là cách pipeline hoạt động từ đầu đến cuối:

1.  **Khởi chạy**: `run.sh` đóng gói thư mục `project` thành `project.zip` và gọi `spark-submit`.
2.  **Chuẩn bị trên Driver**:
    *   `project/main.py` đọc `config.yaml`.
    *   Spark đọc dữ liệu Parquet gốc, **đếm tổng số mẫu (`total_samples`)**, và ghi lại ra một thư mục tạm thời trên HDFS dưới dạng các tệp `part-*.parquet` đã được chuẩn hóa.
    *   Driver thu thập danh sách đường dẫn của các tệp `part-*.parquet` này.
    *   **Phân chia công việc:** Driver tự mình chia danh sách tệp này thành N danh sách nhỏ hơn, mỗi danh sách dành cho một worker.
    *   **Tính toán số bước:** Dựa trên `total_samples` và `global_batch_size`, Driver tính toán chính xác `steps_per_epoch`.
3.  **Huấn luyện phân tán (trên Executor)**:
    *   Mỗi worker nhận danh sách tệp của riêng mình và `steps_per_epoch`.
    *   Worker tạo một `ParquetStreamingDataset` (là một `IterableDataset`). Dataset này được thiết kế để **lặp vô hạn (`itertools.cycle`)** qua các tệp được giao.
    *   Vòng lặp huấn luyện trong `trainer.py` sử dụng **`itertools.islice(dataloader, steps_per_epoch)`**. Đây là điểm mấu chốt: nó "cắt" đúng `steps_per_epoch` batch từ dòng chảy dữ liệu vô hạn.
    *   **Kết quả:** Tất cả các worker chạy chính xác cùng một số bước, đảm bảo chúng đến điểm đồng bộ hóa (`dist.barrier()`) cùng lúc, **tránh hoàn toàn lỗi deadlock**.
4.  **Đánh giá (Validation)**:
    *   Worker `rank=0` tạo một `DataLoader` riêng cho tập validation. `Dataset` cho việc này được cấu hình để chỉ lặp qua dữ liệu một lần và có `__len__` hợp lệ để tương thích.
5.  **Lưu trữ, Báo cáo, Dọn dẹp**: Sau khi huấn luyện xong, Driver sẽ tải lại mô hình tốt nhất, đánh giá trên tập test, và lưu tất cả các kết quả (báo cáo, biểu đồ, model) lên HDFS trước khi dọn dẹp các tệp tạm thời.

## **6. Kết quả đầu ra**

Sau khi chạy xong, bạn sẽ tìm thấy các tệp sau trong thư mục output trên HDFS:

*   `best_<model_type>_model.pth`: Trọng số của mô hình có hiệu suất tốt nhất.
*   `training_history_<model_type>.png`: Biểu đồ loss và accuracy qua các epoch.
*   `training_history_<model_type>.json`: Dữ liệu thô của lịch sử huấn luyện.
*   `final_test_report_<model_type>.json`: Báo cáo chi tiết kết quả trên tập test.
*   `confusion_matrix_<model_type>.png`: Ma trận nhầm lẫn dưới dạng hình ảnh.

## **7. Hướng dẫn Mở rộng Dự án**

Kiến trúc module hóa của dự án giúp cho việc mở rộng trở nên rất đơn giản. Dưới đây là hướng dẫn cho các trường hợp phổ biến.

### **7.1. Cách thêm một Siêu tham số (Hyperparameter) mới**

Giả sử bạn muốn thêm tham số `bidirectional=True` cho mô hình `LSTM`.

1.  **Cập nhật hàm khởi tạo của mô hình:** Chỉnh sửa `project/models/lstm_model.py` để chấp nhận tham số mới và sử dụng nó trong logic của mô hình.
2.  **Thêm tham số vào `config.yaml`:** Thêm `bidirectional: true` vào khối cấu hình của `lstm` trong `model_params`.

Pipeline sẽ tự động nhận diện và sử dụng tham số mới này.

### **7.2. Cách thêm một Mô hình hoàn toàn mới**

Giả sử bạn muốn thêm một mô hình mới có tên là `AttentionGRU`.

1.  **Tạo tệp mô hình mới:** Tạo `project/models/attention_gru_model.py` và định nghĩa lớp `AttentionGRUModel`.
2.  **"Đăng ký" mô hình mới:** Mở `project/models/__init__.py`, import lớp mới và thêm nó vào `model_map`.
3.  **Thêm cấu hình:** Mở `config.yaml`, thêm một khối cấu hình mới cho `attention_gru` trong `model_params` và đặt `training.model_type` thành `attention_gru`.

Bây giờ bạn có thể chạy pipeline để huấn luyện mô hình hoàn toàn mới của mình.
