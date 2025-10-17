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
        # Explicitly use HadoopFileSystem which can handle hdfs:// and alluxio:// schemes
        fs = pyarrow.fs.HadoopFileSystem.from_uri(dir_uri)
        path = urlparse(dir_uri).path

        # Check if directory exists before attempting to delete
        if fs.get_file_info(path).type != pyarrow.fs.FileType.NotFound:
            fs.delete_dir(path)
            print(f"[CLEANUP] Removed directory: {dir_uri}")
        else:
            print(f"[CLEANUP] Directory not found, skipping: {dir_uri}")
    except Exception as e:
        print(f"[CLEANUP] Failed to remove {dir_uri}: {e}")

def upload_log_file(local_log_path, hdfs_output_dir):
    """Uploads a single log file to a specified HDFS directory."""
    if not local_log_path or not os.path.exists(local_log_path):
        print(f"[LOG UPLOAD] Log file not found at {local_log_path}, skipping upload.")
        return

    try:
        print(f"[LOG UPLOAD] Uploading {local_log_path} to {hdfs_output_dir}")
        # Use a temporary directory to upload the single file
        upload_dir = tempfile.mkdtemp()
        shutil.copy(local_log_path, upload_dir)
        
        upload_local_directory_to_hdfs(upload_dir, hdfs_output_dir)
        
        # Clean up the temporary directory
        shutil.rmtree(upload_dir)
        # Clean up the original log file
        os.remove(local_log_path)

    except Exception as e:
        print(f"[LOG UPLOAD] Failed to upload log file {local_log_path}: {e}")