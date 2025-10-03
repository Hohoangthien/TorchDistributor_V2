import os
import json
import tempfile
import shutil
import numpy as np
from urllib.parse import urlparse
import pyarrow.fs

def upload_local_directory_to_hdfs(local_path, hdfs_path):
    try:
        print(f"[HDFS UPLOAD] From '{local_path}' to '{hdfs_path}'")
        parsed_uri = urlparse(hdfs_path)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, port=parsed_uri.port
        )
        hdfs.create_dir(parsed_uri.path, recursive=True)
        for filename in os.listdir(local_path):
            local_file = os.path.join(local_path, filename)
            hdfs_file = os.path.join(parsed_uri.path, filename)
            if os.path.isfile(local_file):
                with open(local_file, "rb") as f_local, hdfs.open_output_stream(
                    hdfs_file
                ) as f_hdfs:
                    f_hdfs.write(f_local.read())
    except Exception as e:
        print(f"[HDFS UPLOAD ERROR] {e}")

def save_and_upload_report(report_data, filename, output_dir):
    local_tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(local_tmp_dir, filename)
    with open(file_path, "w") as f:
        json.dump(
            report_data,
            f,
            indent=2,
            default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else o,
        )
    print(f"[INFO] Saved {filename} locally to: {file_path}")
    if output_dir.startswith("hdfs://"):
        upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
    shutil.rmtree(local_tmp_dir)


def delete_hdfs_directory(dir_uri):
    """Safely deletes a directory on a distributed filesystem (HDFS, Alluxio)."""
    if not dir_uri:
        return
    try:
        fs, path = pyarrow.fs.FileSystem.from_uri(dir_uri)
        if fs.get_file_info(path).type != pyarrow.fs.FileType.NotFound:
            fs.delete_dir(path)
            print(f"[CLEANUP] Removed directory: {dir_uri}")
        else:
            print(f"[CLEANUP] Directory not found, skipping: {dir_uri}")
    except Exception as e:
        print(f"[CLEANUP] Failed to remove {dir_uri}: {e}")