from google.cloud import storage

class GCPStorageClient:
    def __init__(self, bucket_name):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, source_file_path, destination_blob_name):
        """Uploads a file to the bucket."""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        print(f"File {source_file_path} uploaded to {destination_blob_name}.")

    def download_file(self, blob_name, destination_file_path):
        """Downloads a blob from the bucket."""
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(destination_file_path)
        print(f"Blob {blob_name} downloaded to {destination_file_path}.")

    def list_files(self, prefix=None):
        """Lists all the blobs in the bucket that begin with the prefix."""
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        return [blob.name for blob in blobs]

    def delete_file(self, blob_name):
        """Deletes a blob from the bucket."""
        blob = self.bucket.blob(blob_name)
        blob.delete()
        print(f"Blob {blob_name} deleted.")

# Example usage
if __name__ == "__main__":
    bucket_name = "ff_bucket_misc"
    gcp_storage = GCPStorageClient(bucket_name)

    # Upload a file
    gcp_storage.upload_file("tools.py", "tools.py")
    gcp_storage.upload_file("bucket.py", "bucket.py")

    # Download a file
    gcp_storage.download_file("tools.py", "tools_bucket.py")

    # List files in the bucket
    files = gcp_storage.list_files()
    print("Files in bucket:", files)

    # Delete a file
    gcp_storage.delete_file("bucket.py")
