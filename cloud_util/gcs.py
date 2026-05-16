#!/usr/bin/env python
"""
Command-line utility for uploading and downloading files/directories
to and from Google Cloud Storage.

Authentication is via a service-account JSON key file, passed with --key-file
or the GCS_KEY_FILE environment variable.

Sub-commands
------------
upload-file   Upload a single local file to a GCS URI.
upload-dir    Upload a local directory tree to a GCS prefix.
download-file Download a single GCS object to a local directory.
download-dir  Download a GCS prefix (directory) to a local directory.

Examples
--------
# Upload a single file
python gcs.py upload-file model.pt gs://my-bucket/runs/run01/model.pt \\
    --key-file service_account.json

# Upload a directory (safetensors only)
python gcs.py upload-dir ./out/run01 gs://my-bucket/runs/run01/ \\
    --key-file service_account.json --pattern '**/*.safetensors'

# Download a single file into ./data/
python gcs.py download-file gs://my-bucket/data/file.root ./data/ \\
    --key-file service_account.json

# Download all files under a prefix
python gcs.py download-dir gs://my-bucket/runs/run01/ ./local/run01/ \\
    --key-file service_account.json

# Overwrite existing files / blobs
python gcs.py upload-dir ./out gs://my-bucket/out/ --overwrite \\
    --key-file service_account.json
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running this script directly from the cloud_util directory or from the
# repo root, without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "loc_libs"))

try:
    import gcs_utils
except ImportError as exc:
    sys.exit(
        "ERROR: could not import gcs_utils from loc_libs.\n"
        f"       {exc}\n"
        "       Make sure you are running from within the FNO4d repo and that\n"
        "       loc_libs/ is present."
    )


def _resolve_key_file(args) -> str:
    key = getattr(args, "key_file", None) or os.environ.get("GCS_KEY_FILE", "")
    if not key:
        sys.exit(
            "ERROR: GCS key file not specified.\n"
            "       Use --key-file <path> or set the GCS_KEY_FILE environment variable."
        )
    return key


# ---------------------------------------------------------------------------
# sub-command handlers
# ---------------------------------------------------------------------------

def cmd_upload_file(args):
    key = _resolve_key_file(args)
    gcs_utils.upload_gcs_file(
        local_path=args.local_path,
        gs_uri=args.gs_uri,
        key_file=key,
        overwrite=args.overwrite,
    )


def cmd_upload_dir(args):
    key = _resolve_key_file(args)
    gcs_utils.upload_gcs_dir(
        local_dir=args.local_dir,
        gs_prefix=args.gs_prefix,
        key_file=key,
        overwrite=args.overwrite,
        pattern=args.pattern,
    )


def cmd_download_file(args):
    key = _resolve_key_file(args)
    gcs_utils.download_gcs_file(
        gs_uri=args.gs_uri,
        local_dir=args.local_dir,
        key_file=key,
        overwrite=args.overwrite,
        show_progress=None,
    )


def cmd_download_dir(args):
    key = _resolve_key_file(args)
    gcs_utils.download_gcs_dir(
        gs_prefix=args.gs_prefix,
        local_dir=args.local_dir,
        key_file=key,
        overwrite=args.overwrite,
        pattern=args.pattern,
        show_progress=None,
    )


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--key-file", metavar="JSON",
        help="GCS service-account JSON key file "
             "(default: $GCS_KEY_FILE environment variable).",
    )
    parent.add_argument(
        "--overwrite", action="store_true",
        help="Re-upload/re-download even if a matching file already exists.",
    )

    parser = argparse.ArgumentParser(
        prog="gcs.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # upload-file
    p = sub.add_parser(
        "upload-file", parents=[parent],
        help="Upload a single local file to a GCS URI.",
    )
    p.add_argument("local_path", metavar="LOCAL_PATH",
                   help="Path to the local file to upload.")
    p.add_argument("gs_uri", metavar="GS_URI",
                   help="Destination GCS URI, e.g. gs://bucket/path/file.pt")
    p.set_defaults(func=cmd_upload_file)

    # upload-dir
    p = sub.add_parser(
        "upload-dir", parents=[parent],
        help="Upload a local directory tree to a GCS prefix.",
    )
    p.add_argument("local_dir", metavar="LOCAL_DIR",
                   help="Local directory to upload.")
    p.add_argument("gs_prefix", metavar="GS_PREFIX",
                   help="Destination GCS prefix, e.g. gs://bucket/runs/run01/")
    p.add_argument(
        "--pattern", default="**/*", metavar="GLOB",
        help="Glob pattern relative to LOCAL_DIR selecting which files to upload "
             "(default: '**/*' = everything). "
             "Example: '**/*.safetensors'",
    )
    p.set_defaults(func=cmd_upload_dir)

    # download-file
    p = sub.add_parser(
        "download-file", parents=[parent],
        help="Download a single GCS object to a local directory.",
    )
    p.add_argument("gs_uri", metavar="GS_URI",
                   help="Source GCS URI, e.g. gs://bucket/path/file.root")
    p.add_argument("local_dir", metavar="LOCAL_DIR",
                   help="Local directory to write the downloaded file into.")
    p.set_defaults(func=cmd_download_file)

    # download-dir
    p = sub.add_parser(
        "download-dir", parents=[parent],
        help="Download a GCS prefix (directory) to a local directory.",
    )
    p.add_argument("gs_prefix", metavar="GS_PREFIX",
                   help="Source GCS prefix, e.g. gs://bucket/runs/run01/")
    p.add_argument("local_dir", metavar="LOCAL_DIR",
                   help="Local root directory to write files into.")
    p.add_argument(
        "--pattern", default="**/*", metavar="GLOB",
        help="Glob pattern relative to the GCS prefix selecting which blobs to "
             "download (default: '**/*' = everything). "
             "Example: '**/*.safetensors'",
    )
    p.set_defaults(func=cmd_download_dir)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
