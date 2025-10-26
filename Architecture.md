# Tài liệu kiến trúc và hướng dẫn sử dụng

## 1. Tổng quan

Dự án này xây dựng một quy trình (pipeline) hoàn chỉnh để huấn luyện phân tán (distributed training) các mô hình Deep Learning (sử dụng PyTorch) trên các tập dữ liệu lớn. Mục tiêu chính là phát hiện tấn công mạng, với dữ liệu đầu vào đã được tiền xử lý và chuẩn hóa.

Hệ thống được thiết kế để hoạt động trên một cụm hạ tầng Big Data, tận dụng sức mạnh của các công nghệ sau:
- **Lưu trữ phân tán**: HDFS (Hadoop Distributed File System) cho lưu trữ bền vững và Alluxio cho lớp cache tăng tốc.
- **Xử lý dữ liệu lớn**: Apache Spark để tiền xử lý và điều phối.
- **Quản lý tài nguyên**: YARN (Yet Another Resource Negotiator).
- **Huấn luyện phân tán**: `TorchDistributor` của Spark, giúp triển khai PyTorch DDP (DistributedDataParallel).

## 2. Kiến trúc tổng thể

Kiến trúc hệ thống bao gồm sự kết hợp chặt chẽ giữa Spark và PyTorch, được điều phối bởi YARN. Spark không trực tiếp tham gia vào việc tính toán gradient mà đóng vai trò "dàn nhạc trưởng", khởi tạo và quản lý môi trường huấn luyện phân tán cho PyTorch.

```
+--------------------------------------------------------------------------+
| Người dùng (User)                                                        |
|     |                                                                    |
|     | 1. Chạy run.sh / run-client.sh                                     |
|     v                                                                    |
| +----------------------+                                                 |
| | Client Node          |                                                 |
| |   - spark-submit     |<>...............................................+
| |   - config.yaml      |                                                 |
| |   - project.zip      |                                                 |
| +----------------------+                                                 |
|     | 2. YARN yêu cầu tài nguyên                                         |
|     v                                                                    |
| +----------------------------------------------------------------------+ |
| | Cụm YARN (YARN Cluster)                                                | |
| |                                                                      | |
| |   +-------------------------+      +--------------------------------+  | |
| |   | Spark Driver (on YARN)  |----->| Spark Executors (Workers)      |  | |
| |   | - Chạy project/main.py  | 3.   | - TorchDistributor khởi tạo    |  | |
| |   | - Điều phối tác vụ      |      |   môi trường PyTorch DDP       |  | |
| |   +-------------------------+      | - Mỗi executor chạy 1 process  |  | |
| |                                    |   (distributed_trainer.py)     |  | |
| |                                    | - Tải dữ liệu từ Alluxio/HDFS  |  | |
| |                                    | - Đồng bộ gradient (All-Reduce)|  | |
| |                                    +--------------------------------+  | |
| +------------------------------------^---------------------------------+ |
|                                      | 4. Đọc/Ghi dữ liệu               |
|                                      v                                 |
|       +--------------------------+       +--------------------------+    |
|       | Alluxio (Caching Layer)  |<----->| HDFS (Persistent Storage)|    |
|       | - Cache dữ liệu training |       | - Lưu dữ liệu gốc        |    |
|       | - Tăng tốc độ đọc        |       | - Lưu model & results    |    |
|       +--------------------------+       +--------------------------+    |
|                                                                          |
+--------------------------------------------------------------------------+
```

**Luồng hoạt động chính**:
1.  Người dùng thực thi script `run.sh` (cho chế độ cluster) hoặc `run-client.sh` (cho chế độ client) trên một node biên.
2.  `spark-submit` gửi yêu cầu đến YARN để khởi chạy ứng dụng Spark. `project.zip` và `config.yaml` được gửi kèm.
3.  YARN cấp phát tài nguyên và khởi chạy Spark Driver. Spark Driver sau đó yêu cầu các Spark Executor.
4.  Trên mỗi Executor, `TorchDistributor` sẽ thiết lập môi trường huấn luyện PyTorch phân tán (DDP). Mỗi executor sẽ chạy một process training, thực thi logic trong `distributed_trainer.py`.
5.  Các process training cùng nhau đọc dữ liệu từ **Alluxio** (nếu có) hoặc **HDFS**, thực hiện huấn luyện song song, và đồng bộ gradient sau mỗi batch.
6.  Kết quả (model đã huấn luyện, logs, metrics) được ghi lại vào **HDFS**.
7.  Sau khi hoàn tất, các thư mục tạm trên Alluxio/HDFS sẽ được dọn dẹp.

## 3. Hướng dẫn sử dụng

### 3.1. Yêu cầu
- Môi trường Hadoop/Spark đã được cài đặt và cấu hình.
- Python và các thư viện trong `requirements.txt`. Cài đặt bằng lệnh: `pip install -r requirements.txt`.
- `yq` (command-line YAML processor) để đọc file cấu hình. Cài đặt trên Ubuntu: `sudo snap install yq`.

### 3.2. Cấu hình (`config.yaml`, `config-client.yaml`)
- **`config.yaml`**: File cấu hình chính cho môi trường sản xuất (chạy trên cluster).
- **`config-client.yaml`**: File cấu hình cho môi trường phát triển (chạy ở chế độ client), thường có cấu hình tài nguyên thấp hơn.

Các mục chính trong file cấu hình:
- **`active_storage`**: Chọn hệ thống lưu trữ để đọc dữ liệu (`alluxio` hoặc `hdfs`).
- **`data_source`**:
    - `base_path`: Đường dẫn gốc đến dữ liệu.
    - `train_path`, `test_path`: Đường dẫn đến tập huấn luyện và kiểm tra.
    - `label_indexer_path`: Đường dẫn đến model indexer đã lưu.
    - `temp_dir`: Thư mục tạm trên Alluxio/HDFS để lưu các partition dữ liệu được chia nhỏ.
- **`artifact_storage`**:
    - `output_dir`: Đường dẫn trên HDFS để lưu kết quả cuối cùng (model, report, hình ảnh).
- **`training`**:
    - `model_type`: Chọn model để huấn luyện (`gru`, `lstm`, `rnn`, `transformer`).
    - `epochs`, `batch_size`, `learning_rate`: Các siêu tham số huấn luyện.
    - `num_processes`: Số lượng process huấn luyện phân tán. **Phải bằng** `num_executors` trong cấu hình Spark.
- **`spark`**:
    - Chứa các cấu hình cho `spark-submit` như `master`, `deploy_mode`, `num_executors`, `executor_memory`, v.v.
    - Có thể định nghĩa nhiều profile (ví dụ: `spark-cluster`, `spark-client`) để dễ dàng chuyển đổi.
- **`model_params`**:
    - `default`: Các siêu tham số kiến trúc mặc định.
    - Các mục con (`lstm`, `gru`, ...): Ghi đè tham số mặc định cho từng loại model cụ thể.

### 3.3. Thực thi
1.  **Chỉnh sửa `run.sh` / `run-client.sh`**:
    - Cập nhật biến `ALLUXIO_CLIENT_JAR` trỏ đến file JAR client của Alluxio trên máy của bạn.
    - (Tùy chọn) Thay đổi `MODE_SPARK` để chọn profile Spark mong muốn từ `config.yaml`.
2.  **Chạy script**:
    - Để chạy ở chế độ cluster: `bash run.sh`
    - Để chạy ở chế độ client: `bash run-client.sh`
    
    Script sẽ tự động:
    - Đọc file config tương ứng để lấy các tham số Spark.
    - Kiểm tra nếu `alluxio://` được dùng và tự động thêm JARs cần thiết.
    - Nén thư mục `project` thành `project.zip`.
    - Gọi `spark-submit` với đầy đủ các cấu hình.
    - Xóa `project.zip` sau khi hoàn tất.

## 4. Cấu trúc thư mục và giải thích mã nguồn

- **`run.sh`**: Entrypoint để chạy ứng dụng ở chế độ `cluster` trên YARN.
- **`run-client.sh`**: Entrypoint để chạy ứng dụng ở chế độ `client`, thuận tiện cho việc gỡ lỗi.
- **`main.py` (root)**: Một kịch bản Python đơn giản, có thể dùng để khởi chạy `project/main.py` trực tiếp mà không thông qua `spark-submit`, hữu ích cho việc kiểm thử cục bộ.
- **`config.yaml`**: File cấu hình chính cho chế độ cluster.
- **`config-client.yaml`**: File cấu hình cho chế độ client.
- **`requirements.txt`**: Danh sách các thư viện Python cần thiết.
- **`data_prepare/`**: Chứa các notebook cho việc khám phá và tiền xử lý dữ liệu ban đầu.
    - **`nf-uq-nids-v2-preprocess-add-class-weight-spark.ipynb`**: Notebook sử dụng Spark để tiền xử lý tập dữ liệu NF-UQ-NIDS-v2, có thể bao gồm việc tính toán trọng số lớp (class weight).
- **`project/`**: Thư mục mã nguồn chính của ứng dụng Spark.
    - **`main.py`**: **(Chạy trên Spark Driver)**
        - Đây là kịch bản chính điều phối toàn bộ quy trình huấn luyện.
        - **Vai trò**:
            1. Khởi tạo Spark Session.
            2. Đọc `config.yaml`.
            3. **Chuẩn bị dữ liệu**: Sử dụng Spark để đọc và chia nhỏ (re-partition) dữ liệu từ `train_path` và `val_path` thành các file Parquet nhỏ hơn, lưu vào thư mục tạm (`temp_dir`).
            4. Tập hợp tất cả các cấu hình và đường dẫn file vào một dictionary `distributor_args`.
            5. Khởi tạo và chạy `TorchDistributor`, trỏ đến hàm huấn luyện trong `distributed_trainer.py`.
            6. Sau khi training xong, gọi hàm `evaluate_on_test_set` từ `evaluator.py`.
            7. Gọi hàm `delete_hdfs_directory` để dọn dẹp thư mục tạm.
    - **`training/`**: Chứa logic liên quan đến việc huấn luyện và đánh giá model.
        - **`distributed_trainer.py`**: **(Chạy trên Spark Executor/Worker)**. Đây là file cốt lõi của quá trình huấn luyện phân tán.
            - Chứa hàm `training_function` được `TorchDistributor` thực thi trên mỗi process worker.
            - **Vai trò**:
                1. Thiết lập Process Group cho PyTorch DDP (`dist.init_process_group`).
                2. Mỗi process (worker) dựa vào `RANK` của mình để xác định và đọc các file dữ liệu đã được phân chia.
                3. Khởi tạo `DataLoader` để đọc dữ liệu.
                4. Khởi tạo model, bọc nó trong `DistributedDataParallel` (DDP).
                5. Thực hiện vòng lặp training. DDP tự động đồng bộ gradient giữa các worker.
                6. Worker `RANK 0` chịu trách nhiệm thực hiện validation, lưu checkpoint, và ghi lại lịch sử training.
        - **`trainer.py`**: Có thể chứa một lớp `Trainer` cơ bản cho việc huấn luyện trên một node duy nhất, hoặc các logic huấn luyện chung được `distributed_trainer` tái sử dụng.
        - **`evaluator.py`**: Chứa các hàm để đánh giá model trên tập dữ liệu kiểm tra (test set), tính toán các chỉ số (metrics) và tạo báo cáo.
    - **`data/`**: Các module liên quan đến truy cập và xử lý dữ liệu.
        - **`data_loader.py`**: Định nghĩa các lớp `Dataset` của PyTorch, được tối ưu để đọc dữ liệu từ hệ thống file phân tán (HDFS, Alluxio) một cách hiệu quả.
        - **`data_preprocessor.py`**: Chứa hàm `prepare_data_partitions` dùng Spark để chia lại (repartition) dữ liệu đầu vào.
        - **`spark_utils.py`**: Các hàm tiện ích liên quan đến Spark, ví dụ như khởi tạo Spark Session với các cấu hình cụ thể.
    - **`models/`**: Chứa các file định nghĩa kiến trúc mô hình.
        - `base_model.py`: Có thể định nghĩa một lớp model cơ sở.
        - `gru_model.py`, `lstm_model.py`, `rnn_model.py`, `transformer_model.py`: Mỗi file định nghĩa một kiến trúc model cụ thể.
        - `__init__.py`: Chứa hàm `create_model` factory để dễ dàng khởi tạo model dựa trên `model_type` từ config.
    - **`utils/`**: Các module tiện ích dùng chung.
        - **`config.py`**: Đọc và phân tích file `config.yaml`.
        - **`hdfs_utils.py`**: Các hàm tiện ích để tương tác với HDFS/Alluxio (ví dụ: upload, delete).
        - **`logger.py`**: Thiết lập và cấu hình logger cho ứng dụng.
        - **`visualization.py`**: Vẽ và lưu biểu đồ (confusion matrix, training history).
- **`conf/`**: Chứa các file cấu hình cho các thành phần hạ tầng như Hadoop, Spark, Alluxio. Các file này được mount vào container hoặc được môi trường cung cấp sẵn.
