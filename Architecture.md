# Kiến trúc hệ thống Huấn luyện Phân tán

## 1. Tổng quan

Dự án này xây dựng một quy trình (pipeline) hoàn chỉnh để huấn luyện phân tán (distributed training) các mô hình Deep Learning (sử dụng PyTorch) trên các tập dữ liệu lớn. Mục tiêu chính là phát hiện tấn công mạng, với dữ liệu đầu vào đã được tiền xử lý và chuẩn hóa.

Hệ thống được thiết kế để hoạt động trên một cụm hạ tầng Big Data, tận dụng sức mạnh của các công nghệ sau:
- **Lưu trữ phân tán**: HDFS (Hadoop Distributed File System) và Alluxio.
- **Quản lý tài nguyên**: YARN (Yet Another Resource Negotiator).
- **Xử lý dữ liệu**: Apache Spark.
- **Huấn luyện phân tán**: TorchDistributor.

## 2. Kiến trúc tổng thể

Kiến trúc hệ thống bao gồm sự kết hợp chặt chẽ giữa Spark và PyTorch, được điều phối bởi YARN.

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
| |   | - Chạy main.py          | 3.   | - TorchDistributor khởi tạo    |  | |
| |   | - Điều phối tác vụ      |      |   môi trường PyTorch           |  | |
| |   +-------------------------+      | - Mỗi executor chạy 1 process  |  | |
| |                                    |   training                     |  | |
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
3.  YARN cấp phát tài nguyên và khởi chạy Spark Driver trên một node trong cụm. Spark Driver sau đó yêu cầu các Spark Executor.
4.  Trên mỗi Executor, `TorchDistributor` sẽ thiết lập môi trường huấn luyện PyTorch phân tán. Mỗi executor sẽ chạy một process training.
5.  Các process training cùng nhau đọc dữ liệu từ **Alluxio** (nếu có) hoặc **HDFS**, thực hiện huấn luyện song song, và đồng bộ gradient sau mỗi batch.
6.  Kết quả (model đã huấn luyện, logs, metrics) được ghi lại vào **HDFS**.

## 3. Luồng dữ liệu (Data Flow)

### 3.1. Lưu trữ
- **HDFS**: Là nơi lưu trữ chính, dài hạn cho dữ liệu gốc (raw data) và cũng là nơi lưu trữ kết quả cuối cùng (model, report, ...).
- **Alluxio**: Đóng vai trò là một lớp cache (caching layer) nằm giữa HDFS và các ứng dụng tính toán (Spark/PyTorch). Dữ liệu huấn luyện (định dạng Parquet) được lưu trên Alluxio để tăng tốc độ truy cập, giảm thiểu độ trễ I/O trong quá trình training. `config.yaml` chỉ định `active_storage: alluxio`.

### 3.2. Tải và xử lý dữ liệu
- Dữ liệu đầu vào đã được chuẩn hóa và lưu dưới dạng file Parquet với cấu trúc: `[scaled_features, label, weight]`.
- `project/data/data_loader.py` định nghĩa `ParquetStreamingDataset`, một `IterableDataset` của PyTorch.
- Dataset này được thiết kế để đọc dữ liệu trực tiếp từ hệ thống file phân tán (HDFS, Alluxio) một cách hiệu quả. Nó sử dụng `pyarrow.fs.HadoopFileSystem.from_uri` để tự động nhận diện và kết nối đến hệ thống file tương ứng (ví dụ `alluxio://...`).
- Dữ liệu được đọc theo từng batch nhỏ (`iter_batches`) thay vì tải toàn bộ vào bộ nhớ, phù hợp cho các tập dữ liệu cực lớn.

## 4. Luồng Huấn luyện (Training Flow)

### 4.1. Khởi tạo
- `run.sh` là entrypoint chính. Nó đọc file `config.yaml` để lấy các tham số cho `spark-submit` (ví dụ: `num_executors`, `executor_memory`).
- Script này tự động đóng gói thư mục `project` thành `project.zip` để gửi đến các node trong cụm Spark.
- Nó cũng kiểm tra sự tồn tại của Alluxio trong `config.yaml` để tự động thêm các JARs cần thiết vào `spark-submit`.

### 4.2. Điều phối bởi Spark
- `project/main.py` là file python chính được `spark-submit` thực thi trên Spark Driver.
- **Vai trò của Spark ở đây không phải để xử lý logic training**, mà là để **khởi tạo và quản lý môi trường phân tán** cho PyTorch thông qua `TorchDistributor`.
- Spark sẽ đảm bảo rằng `N` (`num_executors`) worker được khởi tạo trên cụm YARN.

### 4.3. Huấn luyện phân tán với TorchDistributor
- `TorchDistributor` (thư viện của Spark) sẽ khởi chạy hàm training trên các worker.
- **Thiết lập môi trường**: Nó tự động thiết lập các biến môi trường cần thiết cho PyTorch DDP (DistributedDataParallel) như `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` trên mỗi worker.
- **Hàm training**: Logic training chính được định nghĩa trong `project/training/trainer.py` và được bọc lại cho môi trường phân tán trong `project/training/distributed_trainer.py`.
- **Mô hình**: Mô hình PyTorch (ví dụ `GRU`, `LSTM` từ `project/models/`) được khởi tạo trên mỗi process và được bọc trong `torch.nn.parallel.DistributedDataParallel`.
- **Phân chia dữ liệu**: `DataLoader` sử dụng `DistributedSampler` để đảm bảo mỗi process training chỉ nhận được một phần (shard) của tập dữ liệu, tránh việc tính toán trùng lặp.
- **Đồng bộ Gradient**: Sau mỗi bước `backward()`, DDP tự động thực hiện phép toán **All-Reduce** để tính trung bình gradient từ tất cả các process và cập nhật lại cho model trên từng process. Điều này đảm bảo rằng tất cả các model con đều học được từ toàn bộ dữ liệu.

### 4.4. Đánh giá và Lưu trữ
- Sau khi quá trình training hoàn tất, model sẽ được đánh giá trên tập test.
- Model tốt nhất (`best_*.pth`), lịch sử training, và các báo cáo kết quả sẽ được lưu vào thư mục `output_dir` đã được định nghĩa trong `config.yaml`, trỏ đến một đường dẫn trên **HDFS**.

## 5. Cấu trúc thư mục

- `main.py`: Entrypoint gốc của project.
- `run.sh`: Script để thực thi ứng dụng Spark.
- `config.yaml`: File cấu hình trung tâm, chứa mọi tham số từ đường dẫn dữ liệu, cấu hình Spark, đến siêu tham số model.
- `project/`: Thư mục mã nguồn chính.
  - `data/`: Chứa logic tải dữ liệu (`data_loader.py`).
  - `models/`: Định nghĩa kiến trúc các mô hình Deep Learning (GRU, LSTM, RNN, Transformer).
  - `training/`: Chứa logic cho việc huấn luyện (`trainer.py`) và huấn luyện phân tán (`distributed_trainer.py`).
  - `utils/`: Các hàm tiện ích (config, logger, hdfs).
- `saves/`: Thư mục (trên máy local) để lưu kết quả nếu chạy ở chế độ client.
- `log/`: Chứa log từ các lần chạy ứng dụng YARN.
