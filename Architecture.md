# Tài liệu kiến trúc và hướng dẫn sử dụng

## 1. Tổng quan

Dự án này xây dựng một quy trình (pipeline) hoàn chỉnh để huấn luyện phân tán (distributed training) các mô hình Deep Learning (sử dụng PyTorch) trên các tập dữ liệu lớn. Mục tiêu chính là phát hiện tấn công mạng, với dữ liệu đầu vào đã được tiền xử lý và chuẩn hóa.

Hệ thống được thiết kế để hoạt động trên một cụm hạ tầng Big Data, tận dụng sức mạnh của các công nghệ sau:
- **Lưu trữ phân tán**: HDFS (Hadoop Distributed File System) cho lưu trữ bền vững và Alluxio cho lớp cache tăng tốc.
- **Quản lý tài nguyên**: YARN (Yet Another Resource Negotiator).
- **Điều phối tác vụ**: Apache Spark.
- **Huấn luyện phân tán**: `TorchDistributor` của Spark, giúp triển khai PyTorch DDP (DistributedDataParallel).

## 2. Kiến trúc tổng thể

Kiến trúc hệ thống bao gồm sự kết hợp chặt chẽ giữa Spark và PyTorch, được điều phối bởi YARN. Spark không trực tiếp tham gia vào việc tính toán gradient mà đóng vai trò "dàn nhạc trưởng", khởi tạo và quản lý môi trường huấn luyện phân tán cho PyTorch.

```
+--------------------------------------------------------------------------+
| Người dùng (User)                                                        |
|     |                                                                    |
|     | 1. Chạy run.sh                                                     |
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
| |                                    |   training (trainer.py)        |  | |
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
1.  Người dùng thực thi script `run.sh` trên một client node.
2.  `spark-submit` gửi yêu cầu đến YARN để khởi chạy ứng dụng Spark. `project.zip` và `config.yaml` được gửi kèm.
3.  YARN cấp phát tài nguyên và khởi chạy Spark Driver. Spark Driver sau đó yêu cầu các Spark Executor.
4.  Trên mỗi Executor, `TorchDistributor` sẽ thiết lập môi trường huấn luyện PyTorch phân tán (DDP). Mỗi executor sẽ chạy một process training.
5.  Các process training cùng nhau đọc dữ liệu từ **Alluxio** (nếu có) hoặc **HDFS**, thực hiện huấn luyện song song, và đồng bộ gradient sau mỗi batch.
6.  Kết quả (model đã huấn luyện, logs, metrics) được ghi lại vào **HDFS**.
7.  Sau khi hoàn tất, các thư mục tạm trên Alluxio/HDFS sẽ được dọn dẹp.

## 3. Hướng dẫn sử dụng

### 3.1. Yêu cầu
- Môi trường Hadoop/Spark đã được cài đặt và cấu hình.
- Python và các thư viện trong `requirements.txt`. Cài đặt bằng lệnh: `pip install -r requirements.txt`.
- `yq` (command-line YAML processor) để đọc file cấu hình. Cài đặt trên Ubuntu: `sudo snap install yq`.

### 3.2. Cấu hình (`config.yaml`)
File `config.yaml` là trung tâm điều khiển của toàn bộ pipeline.

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
    - Có thể định nghĩa nhiều profile (ví dụ: `spark-cluster2`, `spark-client`) để dễ dàng chuyển đổi.
- **`model_params`**:
    - `default`: Các siêu tham số kiến trúc mặc định.
    - Các mục con (`lstm`, `gru`, ...): Ghi đè tham số mặc định cho từng loại model cụ thể.

### 3.3. Thực thi
1.  **Chỉnh sửa `run.sh`**:
    - Cập nhật biến `ALLUXIO_CLIENT_JAR` trỏ đến file JAR client của Alluxio trên máy của bạn.
    - Thay đổi `MODE_SPARK` để chọn profile Spark mong muốn từ `config.yaml` (ví dụ: `spark-client` hoặc `spark-cluster3`).
2.  **Chạy script**:
    ```bash
    bash run.sh
    ```
    Script sẽ tự động:
    - Đọc `config.yaml` để lấy các tham số Spark.
    - Kiểm tra nếu `alluxio://` được dùng và tự động thêm JARs cần thiết.
    - Nén thư mục `project` thành `project.zip`.
    - Gọi `spark-submit` với đầy đủ các cấu hình.
    - Xóa `project.zip` sau khi hoàn tất.

## 4. Cấu trúc thư mục và giải thích mã nguồn

- **`run.sh`**: Entrypoint chính, chịu trách nhiệm đóng gói và gửi ứng dụng Spark.
- **`config.yaml`**: File cấu hình trung tâm.
- **`requirements.txt`**: Danh sách các thư viện Python cần thiết.
- **`project/`**: Thư mục mã nguồn chính.
    - **`main.py`**: **(Chạy trên Spark Driver)**
        - Đây là kịch bản chính điều phối toàn bộ quy trình.
        - **Vai trò**:
            1. Khởi tạo Spark Session.
            2. Đọc `config.yaml`.
            3. **Chuẩn bị dữ liệu**: Sử dụng Spark để đọc và chia nhỏ (re-partition) dữ liệu từ `train_path` và `val_path` thành các file Parquet nhỏ hơn, lưu vào thư mục tạm (`temp_dir`) trên Alluxio. Việc này giúp mỗi worker PyTorch chỉ cần đọc một vài file cụ thể, tránh xung đột I/O.
            4. Tập hợp tất cả các cấu hình và đường dẫn file vào một dictionary `distributor_args`.
            5. Khởi tạo và chạy `TorchDistributor`.
            6. Sau khi training xong, gọi hàm `evaluate_on_test_set` để đánh giá cuối cùng.
            7. Gọi hàm `delete_hdfs_directory` để dọn dẹp các thư mục tạm đã tạo.
    - **`training/trainer.py`**: **(Chạy trên Spark Executor/Worker)**
        - Chứa hàm `training_function` được `TorchDistributor` thực thi trên mỗi process worker.
        - **Vai trò**:
            1. Thiết lập Process Group cho PyTorch DDP (`dist.init_process_group`).
            2. Mỗi process (worker) sẽ được gán một `RANK` (ID định danh).
            3. Dựa vào `RANK`, mỗi worker sẽ biết được cần đọc những file dữ liệu nào từ `distributor_args["files_per_worker"][rank]`.
            4. Khởi tạo `DataLoader` (sử dụng `ParquetStreamingDataset`) để đọc dữ liệu được giao.
            5. Khởi tạo model, bọc nó trong `DistributedDataParallel` (DDP).
            6. Thực hiện vòng lặp training. Sau mỗi bước `backward()`, DDP tự động đồng bộ (all-reduce) gradient giữa các worker.
            7. Worker `RANK 0` chịu trách nhiệm thực hiện validation, lưu checkpoint của model và các file hình ảnh/lịch sử training.
    - **`data/data_loader.py`**:
        - Định nghĩa `ParquetStreamingDataset`, một `IterableDataset` của PyTorch.
        - **Điểm đặc biệt**: Dataset này được thiết kế để đọc dữ liệu trực tiếp từ hệ thống file phân tán (HDFS, Alluxio) một cách hiệu quả. Nó không tải toàn bộ dữ liệu vào bộ nhớ mà đọc theo từng batch nhỏ (`iter_batches`), phù hợp cho các tập dữ liệu cực lớn. Nó sử dụng `pyarrow.fs.HadoopFileSystem.from_uri` để tự động kết nối đến hệ thống file tương ứng.
    - **`data/data_preprocessor.py`**:
        - Chứa hàm `prepare_data_partitions` được gọi từ `main.py`. Hàm này dùng Spark để đọc một file Parquet lớn, `repartition` nó thành số lượng file nhỏ hơn và ghi ra thư mục tạm.
    - **`models/`**:
        - Chứa các file định nghĩa kiến trúc mô hình như `gru_model.py`, `lstm_model.py`.
        - `__init__.py` chứa hàm `create_model` factory để dễ dàng khởi tạo model dựa trên `model_type` từ config.
    - **`utils/`**:
        - `hdfs_utils.py`: Các hàm tiện ích để tương tác với HDFS/Alluxio, ví dụ: `upload_local_directory_to_hdfs` và `delete_hdfs_directory`.
        - `config.py`: Đọc và phân tích file `config.yaml`.
        - `visualization.py`: Vẽ và lưu biểu đồ (confusion matrix, training history).
- **`saves/`**: Thư mục local để lưu kết quả (nếu có).
- **`log/`**: Chứa log từ các lần chạy ứng dụng YARN.