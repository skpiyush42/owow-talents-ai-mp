from google.cloud import storage
import logging
import sys
from google.api_core import exceptions

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ds-cdr-portalreportin-manual-gcs-logger")
logger.setLevel(logging.INFO)

# Class to read files from gcs and export data to gcs
class GcsOperations:
    # gcs client
    gcs_client = storage_client = storage.Client()

    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def get_gcs_blobs(self, file_prefix):
        """
        Function for reading data
        """
        # Accessing bucket
        
        input_bucket = self.gcs_client.bucket(self.bucket_name)

        if not input_bucket.exists():
            logger.error(
                f"GCS bucket does not exist: {input_bucket}")
            raise exceptions.NotFound 
        
        else:
            logger.info(f"GCS bucket exists: {input_bucket}")
        
        logger.info(f"bucket found - {self.bucket_name}")
        blobs = list(input_bucket.list_blobs(prefix=file_prefix))

        return blobs
    
    def check_blob_exists(self, blob_name):
        """
        Checks if a blob exists in the bucket.
        
        :param file_name: file name of the file we want check
        """
        
        # Get the bucket object
        bucket = self.gcs_client.bucket(self.bucket_name)
        
        # Get the blob object
        blob = bucket.blob(blob_name)

        # Check existence
        if blob.exists():
            logger.info(f"File '{blob_name}' exists in bucket '{self.bucket_name}'.")
            return True
        else:
            logger.info(f"File '{blob_name}' does not exist.")
            return False
        
if __name__ == "__main__":
    pass