# **Tài liệu dự án: Pipeline Huấn luyện Phân tán cho các mô hình Deep Learning trên Spark**

*Phiên bản: 2.0 (Kiến trúc Streaming & Tối ưu hóa cho cụm)*
*Công nghệ: Spark, Hadoop, Alluxio, PyTorch*
*Cấu hình cụm tham khảo: 1 Master (Điều phối), 3 Worker (16 Cores, 256GB RAM mỗi node)*

---

## **1. Tổng quan**

Dự án này là một pipeline hoàn chỉnh, mạnh mẽ và có khả năng mở rộng để huấn luyện các mô hình học sâu (tập trung vào các kiến trúc dựa trên RNN) một cách phân tán trên một cụm Big Data (Hadoop/Spark/YARN).

Pipeline được thiết kế theo dạng module hóa, điều khiển bởi cấu hình, và sử dụng kiến trúc nạp dữ liệu kiểu streaming kết hợp với lớp cache Alluxio để xử lý các tập dữ liệu cực lớn một cách hiệu quả.

### **Các tính năng chính:**

*   **Kiến trúc Module hóa:** Mã nguồn được tổ chức sạch sẽ, phân tách rõ ràng các mối quan tâm (dữ liệu, mô hình, huấn luyện, tiện ích).
*   **Hỗ trợ đa mô hình:** Dễ dàng huấn luyện và so sánh các mô hình `RNN`, `LSTM`, `GRU`, và `Transformer`.
*   **Điều khiển bởi Cấu hình:** Toàn bộ pipeline được điều khiển bởi các tệp `config.yaml`, giúp việc thử nghiệm trở nên cực kỳ linh hoạt.
*   **Data Loading hiệu suất cao:** Sử dụng `IterableDataset` để đọc trực tiếp dữ liệu từ hệ thống file phân tán (HDFS, Alluxio) theo kiểu streaming, loại bỏ giới hạn về bộ nhớ.
*   **Tăng tốc I/O với Alluxio:** Tích hợp liền mạch với Alluxio như một lớp cache trong bộ nhớ, giảm đáng kể thời gian đọc dữ liệu từ các epoch thứ hai trở đi.
*   **Huấn luyện phân tán ổn định:** Sử dụng `TorchDistributor` để quản lý vòng đời của các tiến trình huấn luyện PyTorch trên các executor của Spark.
*   **Tự động hóa & Báo cáo:** Tự động chuẩn bị dữ liệu, tạo báo cáo, vẽ biểu đồ và lưu lại các mô hình tốt nhất.
*   **Logging chuyên nghiệp:** Tích hợp hệ thống `logging` chi tiết để dễ dàng theo dõi và gỡ lỗi.

## **2. Cấu trúc dự án**

Dự án được tổ chức theo cấu trúc giúp dễ dàng quản lý và mở rộng:

```
TorchDistributor_V1
├─ Architecture.md
├─ README.md
├─ conf
│  ├─ alluxio
│  │  ├─ alluxio-site.properties
│  │  ├─ masters
│  │  └─ workers
│  ├─ hadoop
│  │  ├─ core-site.xml
│  │  ├─ hadoop-env.sh
│  │  ├─ hdfs-site.xml
│  │  ├─ httpfs-log4j.properties
│  │  ├─ mapred-site.xml
│  │  ├─ workers
│  │  └─ yarn-site.xml
│  └─ spark
│     ├─ spark-defaults.conf
│     ├─ spark-env.sh
│     ├─ spark-env.sh.template
│     ├─ spark-env.sh.worker
│     └─ workers
├─ config-client.yaml
├─ config.yaml
├─ data_prepare
│  └─ nf-uq-nids-v2-preprocess-add-class-weight-spark.ipynb
├─ main.py
├─ project
│  ├─ data
│  │  ├─ data_loader.py
│  │  ├─ data_preprocessor.py
│  │  ├─ preprocess.IPYNB
│  │  └─ spark_utils.py
│  ├─ main.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ base_model.py
│  │  ├─ gru_model.py
│  │  ├─ lstm_model.py
│  │  ├─ rnn_model.py
│  │  └─ transformer_model.py
│  ├─ training
│  │  ├─ __init__.py
│  │  ├─ distributed_trainer.py
│  │  ├─ evaluator.py
│  │  └─ trainer.py
│  └─ utils
│     ├─ __init__.py
│     ├─ config.py
│     ├─ hdfs_utils.py
│     ├─ logger.py
│     └─ visualization.py
├─ requirements.txt
├─ run-client.sh
└─ run.sh

```

## **3. Cài đặt và Thiết lập**

1.  **Môi trường Python:**
    *   Đảm bảo một môi trường Python (ví dụ: Conda, venv) đã được cài đặt và kích hoạt trên tất cả các node của cụm.

2.  **Cài đặt các gói phụ thuộc:**
    *   Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

## **4. Hướng dẫn sử dụng và Cấu hình**

### **4.1. Tệp cấu hình `config.yaml` và `config-client.yaml`**

Đây là trung tâm điều khiển của pipeline. Mọi thay đổi có thể được thực hiện ở đây mà không cần sửa mã nguồn.

*   **`config.yaml`**: Dùng khi chạy với `run.sh` ở chế độ cluster. Chứa nhiều profile (`spark`, `spark-cluster2`,...) cho các kịch bản tài nguyên khác nhau.
*   **`config-client.yaml`**: Dùng khi chạy với `run-client.sh` ở chế độ client, thường có cấu hình nhẹ hơn để dễ gỡ lỗi.
*   **`spark`**: Chứa các cấu hình để khởi chạy công việc Spark.
*   **`data_source` / `artifact_storage`**: Định nghĩa các đường dẫn đến dữ liệu và kết quả trên HDFS/Alluxio.
*   **`training`**: Chứa các siêu tham số huấn luyện.
*   **`model_params`**: Cho phép định nghĩa siêu tham số kiến trúc cho từng mô hình.

### **4.2. Các Chiến lược Cấu hình Spark (Qua `config.yaml`)**

Điểm mạnh của dự án là khả năng định nghĩa và chuyển đổi giữa các chiến lược phân bổ tài nguyên Spark một cách linh hoạt ngay trong `config.yaml`. Script `run.sh` sẽ đọc một profile được chỉ định và xây dựng lệnh `spark-submit` tương ứng.

Cụm có 3 worker, mỗi worker 16 cores, tổng cộng **48 cores** khả dụng. Dưới đây là phân tích các chiến lược đã định nghĩa:

**Chiến lược 1: Nhiều Executor, kích thước vừa phải (Profile `spark`)**

```yaml
spark:
  num_executors: 12
  executor_cores: 4
  executor_memory: 32G
```
*   **Phân tích:** Yêu cầu 12 executor, mỗi executor 4 cores (`12 * 4 = 48` cores). Mỗi worker node sẽ chạy `12 / 3 = 4` executor.
*   **Đánh giá:** Tăng độ song song ở mức executor, nhưng có thể tăng chi phí giao tiếp và gây xung đột I/O trên worker node.

**Chiến lược 2: Ít Executor, kích thước lớn (Profile `spark-cluster2`)**

```yaml
spark-cluster2:
  num_executors: 6
  executor_cores: 8
  executor_memory: 50G
```
*   **Phân tích:** Yêu cầu 6 executor, mỗi executor 8 cores (`6 * 8 = 48` cores). Mỗi worker node sẽ chạy `6 / 3 = 2` executor.
*   **Đánh giá:** Một sự cân bằng tốt giữa độ song song và tài nguyên cho mỗi executor. Thường là lựa chọn tốt cho các tác vụ ML.

### **4.3. Chạy Pipeline**

1.  **Chọn Profile Spark (Tùy chọn):** Mở `run.sh` và thay đổi biến `MODE_SPARK`. Script này sử dụng biến đó để đọc profile tương ứng từ `config.yaml`.
    
    ```bash
    # Ví dụ bên trong file run.sh
    #!/bin/bash
    
    # CHỈNH SỬA DÒNG NÀY ĐỂ CHỌN PROFILE
    MODE_SPARK="spark-cluster2"
    
    # Phần còn lại của script sẽ dùng yq để đọc cấu hình từ profile này
    # Ví dụ: NUM_EXECUTORS=$(yq e ".${MODE_SPARK}.num_executors" config.yaml)
    ```

2.  **Cấp quyền thực thi cho script:** (Chỉ cần làm một lần)
    ```bash
    chmod +x run.sh run-client.sh
    ```
3.  **Chạy pipeline:**
    *   Để chạy trên cụm YARN (chế độ production):
        ```bash
        bash run.sh
        ```
    *   Để chạy ở chế độ client (Spark Driver chạy tại máy submit, tiện cho gỡ lỗi):
        ```bash
        bash run-client.sh
        ```

## **5. Luồng hoạt động chi tiết**

1.  **Khởi chạy**: `run.sh` hoặc `run-client.sh` đọc profile Spark, đóng gói thư mục `project` và gọi `spark-submit`.
2.  **Chuẩn bị trên Driver**: `project/main.py` được thực thi trên Spark Driver. Nó đọc cấu hình, `repartition` dữ liệu và phân chia danh sách các file dữ liệu cho từng executor.
3.  **Huấn luyện phân tán**: `TorchDistributor` khởi tạo các tiến trình PyTorch trên mỗi Executor. Mỗi tiến trình đọc phần dữ liệu được giao và thực hiện huấn luyện. PyTorch DDP tự động đồng bộ gradient giữa các tiến trình.
4.  **Đánh giá**: Worker `rank=0` thực hiện đánh giá trên tập validation sau mỗi epoch và lưu lại checkpoint.
5.  **Lưu trữ và Dọn dẹp**: Sau khi huấn luyện, Driver lưu các kết quả cuối cùng lên HDFS và dọn dẹp tài nguyên tạm.

## **6. Kết quả đầu ra**

Kết quả được lưu trong thư mục `artifact_storage.output_dir` trên HDFS, bao gồm:
*   `best_<model_type>_model.pth`: Trọng số của mô hình tốt nhất.
*   `training_history_<model_type>.png`: Biểu đồ loss và accuracy.
*   `training_history_<model_type>.json`: Dữ liệu lịch sử huấn luyện.
*   `final_test_report_<model_type>.json`: Báo cáo kết quả trên tập test.
*   `confusion_matrix_<model_type>.png`: Ma trận nhầm lẫn.

## **7. Hướng dẫn Mở rộng Dự án**

Kiến trúc module hóa giúp việc mở rộng trở nên đơn giản.

### **7.1. Thêm Siêu tham số mới**

1.  **Cập nhật Model:** Sửa file model trong `project/models/` để chấp nhận tham số mới.
2.  **Cập nhật Config:** Thêm tham số vào `config.yaml` trong khối `model_params` tương ứng.

### **7.2. Thêm Mô hình mới**

1.  **Tạo file Model:** Tạo file `.py` mới trong `project/models/`.
2.  **Đăng ký Model:** Mở `project/models/__init__.py`, import và thêm model vào `model_map`.
3.  **Thêm Config:** Thêm khối cấu hình cho model mới trong `model_params` của `config.yaml`.

## **8. Gỡ lỗi và Theo dõi (Debugging & Monitoring)**

Khi một tác vụ Spark thất bại, việc tìm kiếm log là bước quan trọng nhất.

*   **Spark History Server:** Truy cập giao diện web của Spark History Server (địa chỉ được cấu hình trong `mapred-site.xml`) để xem UI của các ứng dụng đã hoàn thành, bao gồm các giai đoạn (stages), tác vụ (tasks) và log.

*   **Dòng lệnh YARN:** Cách nhanh nhất để lấy log của một ứng dụng thất bại là sử dụng ID của nó.
    ```bash
    # Lấy ID ứng dụng từ output của run.sh hoặc từ YARN ResourceManager UI
    APPLICATION_ID="application_167..."
    
    # Lệnh để xem toàn bộ log
    yarn logs -applicationId $APPLICATION_ID
    ```

*   **Tìm kiếm trong log:** Log của PyTorch (từ các lệnh `print` hoặc `logging`) sẽ nằm trong `stdout`/`stderr` của các executor. Khi dùng lệnh `yarn logs`, hãy tìm kiếm các thông báo lỗi từ PyTorch hoặc các thông điệp log đã thêm vào mã nguồn.
