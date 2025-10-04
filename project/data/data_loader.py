import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from urllib.parse import urlparse
import pyarrow.parquet as pq
import pyarrow.fs
import itertools
import math

class ParquetStreamingDataset(IterableDataset):
    """A robust, universal iterable dataset for streaming from Parquet files.
    - In training mode, it cycles infinitely over the data.
    - In evaluation mode, it iterates once.
    - It always provides a __len__ method to ensure full compatibility.
    """
    def __init__(self, file_paths, batch_size, num_samples, is_training=False):
        super(ParquetStreamingDataset, self).__init__()
        self.file_paths = file_paths if file_paths else []
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.is_training = is_training
        self.hdfs = None

    def __len__(self):
        """Returns the number of batches the dataset will produce in one pass."""
        if not self.batch_size:
            return 0
        if self.num_samples == 0:
            return 1
        return math.ceil(self.num_samples / self.batch_size)

    def _init_filesystem(self):
        """Initializes a generic HDFS-compatible filesystem from the file URI...."""
        if self.hdfs or not self.file_paths:
            return

        uri = self.file_paths[0]
        try:
            self.hdfs = pyarrow.fs.HadoopFileSystem.from_uri(uri)
        except Exception as e:
            print(f"[DATASET_ERROR] Failed to initialize filesystem for URI '{uri}': {e}")

    def __iter__(self):
        self._init_filesystem()
        
        iterator = itertools.cycle(self.file_paths) if self.is_training else iter(self.file_paths)
        
        for file_path in iterator:
            try:
                parquet_file = pq.ParquetFile(file_path, filesystem=self.hdfs)
                for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                    pydict = batch.to_pydict()
                    yield (
                        torch.tensor(pydict['scaled_features'], dtype=torch.float32),
                        torch.tensor(pydict['label'], dtype=torch.long),
                        torch.tensor(pydict['weight'], dtype=torch.float32),
                    )
            except Exception as e:
                if self.is_training:
                    print(f"[DATASET_WARNING] Skipping problematic file {file_path}: {e}")
                    continue
                else:
                    print(f"[DATASET_ERROR] Failed to read evaluation file {file_path}: {e}")
                    break # Stop on error during evaluation

def create_pytorch_dataloader(file_paths, batch_size, num_samples, is_training=False):
    """Creates a DataLoader with the universal ParquetStreamingDataset."""
    if not file_paths or not file_paths[0]:
        return None
    dataset = ParquetStreamingDataset(file_paths, batch_size, num_samples, is_training)
    return DataLoader(dataset, batch_size=None, num_workers=0)
