"""
Google Cloud Storage utilities for FNO4d data I/O.

This module is an optional extension — it is only functional when the GCS
optional dependencies are installed:

    pip install -e './loc_libs[gcs]'
    # or:
    pip install -r requirements-gcs.txt   # gcsfs, google-cloud-storage

Two usage patterns are supported:

Streaming (fsspec / uproot reads gs:// URIs directly):
    opts = get_gcs_storage_options(key_file)
    data = read_3d_data("gs://bucket/file.root", storage_options=opts)

Download-then-read (copy to local disk first):
    local_paths = download_gcs_files(gs_uris, local_dir, key_file)
    data = read_3d_data(local_paths[0])

GCS authentication is provided exclusively via a service-account JSON key file.
"""

import fnmatch
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# optional-dependency guards
# ---------------------------------------------------------------------------
try:
    import gcsfs as _gcsfs  # noqa: F401
    _GCSFS_AVAILABLE = True
except ImportError:
    _GCSFS_AVAILABLE = False

try:
    from google.cloud import storage as _gcs  # noqa: F401
    from google.oauth2 import service_account as _sa  # noqa: F401
    _GCS_STORAGE_AVAILABLE = True
except ImportError:
    _GCS_STORAGE_AVAILABLE = False


def _require_gcsfs():
    if not _GCSFS_AVAILABLE:
        raise ImportError(
            "gcsfs is required for GCS streaming.\n"
            "Install with:  pip install -e './loc_libs[gcs]'\n"
            "           or: pip install -r requirements-gcs.txt"
        )


def _require_gcs_storage():
    if not _GCS_STORAGE_AVAILABLE:
        raise ImportError(
            "google-cloud-storage is required for GCS downloads.\n"
            "Install with:  pip install -e './loc_libs[gcs]'\n"
            "           or: pip install -r requirements-gcs.txt"
        )


# ---------------------------------------------------------------------------
# streaming helpers
# ---------------------------------------------------------------------------

def get_gcs_storage_options(key_file: str) -> dict:
    """Return *storage_options* for uproot/fsspec to stream gs:// URIs.

    Pass the result as the ``storage_options`` keyword argument of
    ``read_3d_data``.  uproot forwards it to the underlying ``gcsfs``
    filesystem via fsspec.

    Args:
        key_file: Path to a GCS service-account JSON key file.

    Returns:
        Dict suitable for ``uproot.open(uri, **storage_options)``.

    Raises:
        ImportError: if gcsfs is not installed.
        FileNotFoundError: if *key_file* does not exist.
    """
    _require_gcsfs()
    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")
    return {"token": str(key_path)}


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------

def parse_gcs_uri(gs_uri: str):
    """Split a ``gs://bucket/path`` URI into ``(bucket, blob_path)``."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {gs_uri!r}")
    without_scheme = gs_uri[len("gs://"):]
    bucket, _, blob_path = without_scheme.partition("/")
    return bucket, blob_path


def download_gcs_file(gs_uri: str, local_dir: str, key_file: str,
                      overwrite: bool = False) -> str:
    """Download a single GCS object to *local_dir*.

    The destination filename is the basename of the blob path.  If the local
    file already exists and its byte-size matches the remote blob, the
    download is skipped (unless *overwrite* is True).

    Args:
        gs_uri:    Source URI, e.g. ``gs://my-bucket/data/file.root``.
        local_dir: Local directory to write the file into (created if absent).
        key_file:  Path to a GCS service-account JSON key file.
        overwrite: Always download, even if a matching local file exists.

    Returns:
        Absolute path (str) of the downloaded (or already-cached) file.

    Raises:
        ImportError: if google-cloud-storage is not installed.
        FileNotFoundError: if *key_file* does not exist.
    """
    _require_gcs_storage()

    from google.cloud import storage
    from google.oauth2 import service_account

    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")

    creds = service_account.Credentials.from_service_account_file(str(key_path))
    client = storage.Client(credentials=creds, project=creds.project_id)

    bucket_name, blob_path = parse_gcs_uri(gs_uri)
    filename = Path(blob_path).name
    local_path = Path(local_dir) / filename

    if not overwrite and local_path.exists():
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.reload()
        if local_path.stat().st_size == blob.size:
            print(f"[GCS] {filename} already cached at {local_path}"
                  f"  ({blob.size / 1024**2:.1f} MiB)  — skipping download")
            return str(local_path)

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"[GCS] downloading gs://{bucket_name}/{blob_path}")
    print(f"[GCS]          → {local_path}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    size_mib = local_path.stat().st_size / 1024**2
    print(f"[GCS] done  ({size_mib:.1f} MiB)")
    return str(local_path)


def download_gcs_files(gs_uris: list, local_dir: str, key_file: str,
                       overwrite: bool = False) -> list:
    """Download multiple GCS objects to *local_dir*.

    Calls :func:`download_gcs_file` for each URI in order.

    Returns:
        List of local paths (str) in the same order as *gs_uris*.
    """
    return [
        download_gcs_file(uri, local_dir, key_file, overwrite=overwrite)
        for uri in gs_uris
    ]


def download_gcs_prefix(gs_prefix: str, local_dir: str, key_file: str,
                        overwrite: bool = False,
                        preserve_structure: bool = True) -> list:
    """Download all blobs under a GCS prefix (directory) to *local_dir*.

    Equivalent to ``gsutil -m cp -r gs://bucket/prefix/ local_dir/``.

    Args:
        gs_prefix:          GCS URI ending with ``/`` or not, e.g.
                            ``gs://my-bucket/data/`` or ``gs://my-bucket/data``.
        local_dir:          Local root directory to write files into.
        key_file:           Path to a GCS service-account JSON key file.
        overwrite:          Always download, even if a matching local file exists.
        preserve_structure: If True (default), recreate the prefix-relative
                            subdirectory tree under *local_dir*.
                            If False, all files land flat in *local_dir*
                            (basename only — collisions are silently overwritten).

    Returns:
        List of local paths (str) for every file downloaded or already cached,
        in the order returned by the GCS list API.

    Raises:
        ImportError: if google-cloud-storage is not installed.
        FileNotFoundError: if *key_file* does not exist.
        ValueError: if *gs_prefix* is not a ``gs://`` URI.
    """
    _require_gcs_storage()

    from google.cloud import storage
    from google.oauth2 import service_account

    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")

    creds = service_account.Credentials.from_service_account_file(str(key_path))
    client = storage.Client(credentials=creds, project=creds.project_id)

    bucket_name, blob_prefix = parse_gcs_uri(gs_prefix)
    # normalise: strip trailing slash so prefix matching is consistent
    blob_prefix = blob_prefix.rstrip("/")

    blobs = list(client.list_blobs(bucket_name, prefix=blob_prefix))
    if not blobs:
        print(f"[GCS] no blobs found under gs://{bucket_name}/{blob_prefix}")
        return []

    print(f"[GCS] found {len(blobs)} object(s) under gs://{bucket_name}/{blob_prefix}")
    local_paths = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # skip directory placeholder objects

        if preserve_structure:
            # strip the prefix, keeping the relative sub-path
            rel = blob.name[len(blob_prefix):].lstrip("/")
            local_path = Path(local_dir) / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            local_path = Path(local_dir) / Path(blob.name).name
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        if not overwrite and local_path.exists():
            if local_path.stat().st_size == blob.size:
                print(f"[GCS] {blob.name}  →  cached ({blob.size / 1024**2:.1f} MiB)")
                local_paths.append(str(local_path))
                continue

        print(f"[GCS] {blob.name}  →  {local_path}  ({blob.size / 1024**2:.1f} MiB)")
        blob.download_to_filename(str(local_path))
        local_paths.append(str(local_path))

    print(f"[GCS] prefix sync complete — {len(local_paths)} file(s) in {local_dir}")
    return local_paths


# ---------------------------------------------------------------------------
# upload helpers
# ---------------------------------------------------------------------------

def upload_gcs_file(local_path: str, gs_uri: str, key_file: str,
                    overwrite: bool = False) -> str:
    """Upload a single local file to a GCS destination URI.

    If *overwrite* is False (default) and a blob already exists at *gs_uri*
    with the same byte-size as the local file, the upload is skipped.

    Args:
        local_path: Path to the local file to upload.
        gs_uri:     Destination URI, e.g. ``gs://my-bucket/checkpoints/model.pt``.
        key_file:   Path to a GCS service-account JSON key file.
        overwrite:  Always upload, even if a matching remote blob exists.

    Returns:
        The destination URI (str) — *gs_uri* unchanged.

    Raises:
        ImportError: if google-cloud-storage is not installed.
        FileNotFoundError: if *local_path* or *key_file* does not exist.
    """
    _require_gcs_storage()

    from google.cloud import storage
    from google.oauth2 import service_account

    local = Path(local_path)
    if not local.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")

    creds = service_account.Credentials.from_service_account_file(str(key_path))
    client = storage.Client(credentials=creds, project=creds.project_id)

    bucket_name, blob_path = parse_gcs_uri(gs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not overwrite:
        # check remote size without downloading
        if blob.exists():
            blob.reload()
            if blob.size == local.stat().st_size:
                print(f"[GCS] {local.name}  already uploaded ({blob.size / 1024**2:.1f} MiB)"
                      f"  — skipping")
                return gs_uri

    size_mib = local.stat().st_size / 1024**2
    print(f"[GCS] uploading {local}  →  {gs_uri}  ({size_mib:.1f} MiB)")
    blob.upload_from_filename(str(local))
    print(f"[GCS] done")
    return gs_uri


def upload_gcs_dir(local_dir: str, gs_prefix: str, key_file: str,
                   overwrite: bool = False,
                   pattern: str = "**/*") -> list:
    """Upload a local directory tree to a GCS prefix.

    Equivalent to ``gsutil -m cp -r local_dir/ gs://bucket/prefix/``.

    The prefix-relative path of each file is preserved under *gs_prefix*:

    .. code-block:: text

        local_dir/best_model.safetensors  →  gs_prefix/best_model.safetensors
        local_dir/checkpoint_10/model.pt  →  gs_prefix/checkpoint_10/model.pt

    Args:
        local_dir:  Local directory whose contents are uploaded.
        gs_prefix:  Destination GCS prefix, e.g. ``gs://my-bucket/runs/run01/``.
        key_file:   Path to a GCS service-account JSON key file.
        overwrite:  Always upload, even if a matching remote blob exists.
        pattern:    Glob pattern (relative to *local_dir*) selecting which files
                    to upload.  Default ``"**/*"`` matches everything recursively.
                    Examples:
                      ``"*.safetensors"``        — only safetensors in the root
                      ``"**/*.safetensors"``      — safetensors anywhere in tree
                      ``"**/*.{safetensors,json}"`` — safetensors + JSON files
                      ``"checkpoint_*/**/*"``     — only checkpoint subdirs

    Returns:
        List of destination GCS URIs (str) for every file uploaded or
        already present, in filesystem traversal order.

    Raises:
        ImportError: if google-cloud-storage is not installed.
        FileNotFoundError: if *local_dir* or *key_file* does not exist.
        ValueError: if *gs_prefix* is not a ``gs://`` URI.
    """
    _require_gcs_storage()

    from google.cloud import storage
    from google.oauth2 import service_account

    local_root = Path(local_dir)
    if not local_root.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")

    creds = service_account.Credentials.from_service_account_file(str(key_path))
    client = storage.Client(credentials=creds, project=creds.project_id)

    bucket_name, blob_prefix = parse_gcs_uri(gs_prefix)
    blob_prefix = blob_prefix.rstrip("/")
    bucket = client.bucket(bucket_name)

    # collect files matching the glob pattern (directories excluded)
    files = [p for p in local_root.glob(pattern) if p.is_file()]
    if not files:
        print(f"[GCS] no files matched pattern '{pattern}' in {local_dir}")
        return []

    print(f"[GCS] uploading {len(files)} file(s) from {local_dir}"
          f"  →  gs://{bucket_name}/{blob_prefix}/")

    gs_uris = []
    for local_file in sorted(files):
        rel = local_file.relative_to(local_root)
        dest_blob_path = f"{blob_prefix}/{rel}" if blob_prefix else str(rel)
        gs_uri = f"gs://{bucket_name}/{dest_blob_path}"

        blob = bucket.blob(dest_blob_path)
        size_mib = local_file.stat().st_size / 1024**2

        if not overwrite and blob.exists():
            blob.reload()
            if blob.size == local_file.stat().st_size:
                print(f"[GCS]   {rel}  already uploaded ({size_mib:.1f} MiB)  — skipping")
                gs_uris.append(gs_uri)
                continue

        print(f"[GCS]   {rel}  →  {gs_uri}  ({size_mib:.1f} MiB)")
        blob.upload_from_filename(str(local_file))
        gs_uris.append(gs_uri)

    print(f"[GCS] upload complete — {len(gs_uris)} file(s) at gs://{bucket_name}/{blob_prefix}/")
    return gs_uris


def _rel_matches(rel: str, pattern: str) -> bool:
    """Return True if the relative POSIX path *rel* matches *pattern*.

    Handles the three practically relevant cases on Python 3.9+:
      ``"**/*"``          — match everything (fast path, the default)
      ``"*.ext"``         — match by extension at any depth
      ``"**/*.ext"``      — same (strips the ``**/`` prefix and checks basename)
      ``"subdir/**/*"``   — match a named subtree (uses Path.match, 3.12+ for **)
    """
    if pattern == "**/*":
        return True
    p = Path(rel)
    # For "**/<tail>" patterns: match the basename against <tail>.
    # This gives correct results on Python 3.9–3.11 where Path.match() does not
    # treat ** as "any number of components".
    if pattern.startswith("**/"):
        tail = pattern[3:]
        if fnmatch.fnmatch(p.name, tail):
            return True
    # Path.match() handles simple patterns ("*.ext", "dir/sub/*.ext") on all
    # versions, and full ** semantics on Python 3.12+.
    return p.match(pattern)


def download_gcs_dir(gs_prefix: str, local_dir: str, key_file: str,
                     overwrite: bool = False,
                     pattern: str = "**/*") -> list:
    """Download a GCS prefix (directory) to a local directory.

    Symmetric counterpart to :func:`upload_gcs_dir`.  The prefix-relative
    path of every matching blob is preserved under *local_dir*:

    .. code-block:: text

        gs_prefix/best_model.safetensors  →  local_dir/best_model.safetensors
        gs_prefix/checkpoint_10/model.pt  →  local_dir/checkpoint_10/model.pt

    Equivalent to ``gsutil -m cp -r gs://bucket/prefix/ local_dir/``.

    Args:
        gs_prefix:  Source GCS URI, e.g. ``gs://my-bucket/runs/run01/``.
        local_dir:  Local root directory to write files into (created if absent).
        key_file:   Path to a GCS service-account JSON key file.
        overwrite:  Always download, even if a matching local file exists.
        pattern:    Glob pattern (relative to the prefix) selecting which blobs
                    to download.  Default ``"**/*"`` downloads everything.
                    Examples::

                      "*.safetensors"       — root-level safetensors only
                      "**/*.safetensors"    — safetensors at any depth
                      "**/*.json"           — JSON files at any depth
                      "checkpoint_*/**/*"   — only checkpoint subdirs (3.12+)

                    ``**`` across multiple path components requires Python 3.12+
                    except for the leading ``**/`` form, which is handled on
                    all supported Python versions.

    Returns:
        List of local paths (str) for every file downloaded or already cached,
        in the order returned by the GCS list API.

    Raises:
        ImportError: if google-cloud-storage is not installed.
        FileNotFoundError: if *key_file* does not exist.
        ValueError: if *gs_prefix* is not a ``gs://`` URI.
    """
    _require_gcs_storage()

    from google.cloud import storage
    from google.oauth2 import service_account

    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")

    creds = service_account.Credentials.from_service_account_file(str(key_path))
    client = storage.Client(credentials=creds, project=creds.project_id)

    bucket_name, blob_prefix = parse_gcs_uri(gs_prefix)
    blob_prefix = blob_prefix.rstrip("/")

    all_blobs = [b for b in client.list_blobs(bucket_name, prefix=blob_prefix)
                 if not b.name.endswith("/")]

    if not all_blobs:
        print(f"[GCS] no blobs found under gs://{bucket_name}/{blob_prefix}/")
        return []

    # filter by pattern applied to the prefix-relative path
    matched = []
    for blob in all_blobs:
        rel = blob.name[len(blob_prefix):].lstrip("/")
        if _rel_matches(rel, pattern):
            matched.append((blob, rel))

    if not matched:
        print(f"[GCS] no blobs matched pattern '{pattern}' under"
              f" gs://{bucket_name}/{blob_prefix}/")
        return []

    print(f"[GCS] downloading {len(matched)} of {len(all_blobs)} object(s)"
          f" from gs://{bucket_name}/{blob_prefix}/  →  {local_dir}/")

    local_paths = []
    for blob, rel in matched:
        local_path = Path(local_dir) / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        size_mib = blob.size / 1024**2

        if not overwrite and local_path.exists():
            if local_path.stat().st_size == blob.size:
                print(f"[GCS]   {rel}  already cached ({size_mib:.1f} MiB)  — skipping")
                local_paths.append(str(local_path))
                continue

        print(f"[GCS]   {rel}  ({size_mib:.1f} MiB)")
        blob.download_to_filename(str(local_path))
        local_paths.append(str(local_path))

    print(f"[GCS] download complete — {len(local_paths)} file(s) in {local_dir}/")
    return local_paths


# ---------------------------------------------------------------------------
# availability check (useful in scripts to give an early error)
# ---------------------------------------------------------------------------

def check_gcs_dependencies(require_storage: bool = False):
    """Raise ImportError if required GCS packages are missing.

    Args:
        require_storage: Also check for ``google-cloud-storage`` (needed for
                         downloads).  Default only checks ``gcsfs`` (needed
                         for streaming).
    """
    _require_gcsfs()
    if require_storage:
        _require_gcs_storage()
