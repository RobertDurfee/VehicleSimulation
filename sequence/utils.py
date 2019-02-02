from google.cloud import storage
import re


gcs_url_re = re.compile(r"^(gs:\/\/)(?P<bucket_name>[a-z._\-]+)\/(?P<object_name>.+)$")

def copy_to_gcs(local_file, remote_file):
    """Copy a file from local storage to Google Cloud Storage.

    Args:
        local_file (str): Path to local file source.
        remote_file (str): Path to remote file destination.

    """
    match = gcs_url_re.match(remote_file)
    if not match:
        raise ValueError('\'' + remote_file + '\' is a malformed GCS URL.')

    bucket_name = match['bucket_name']
    object_name = match['object_name'].replace('\\', '/')

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)

    blob.upload_from_filename(local_file)


def copy_from_gcs(remote_file, local_file):
    """Copy a file from remote Google Cloud Storage.

    Args:
        remote_file (str): Path to remote file source.
        local_file (str): Path to local file destination.

    """
    match = gcs_url_re.match(remote_file)
    if not match:
        raise ValueError('\'' + remote_file + '\' is a malformed GCS URL.')

    bucket_name = match['bucket_name']
    object_name = match['object_name'].replace('\\', '/')

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)

    blob.download_to_filename(local_file)
