#!/usr/bin/env python
"""
Command-line utility for uploading and downloading files/directories
to and from Google Cloud Storage.

Authentication is via a service-account JSON key file, passed with --key-file
or the GCS_KEY_FILE environment variable.

Sub-commands
------------
ls            List objects under a GCS prefix with size and modification time.
upload-file   Upload a single local file to a GCS URI.
upload-dir    Upload a local directory tree to a GCS prefix.
download-file Download a single GCS object to a local directory.
download-dir  Download a GCS prefix (directory) to a local directory.

Examples
--------
# List top-level items (files + subdirectories)
python gcs.py ls gs://my-bucket/runs/run01/ --key-file service_account.json

# List all objects recursively
python gcs.py ls gs://my-bucket/runs/run01/ -r --key-file service_account.json

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


def _fmt_size(size_bytes: int) -> str:
    for unit, factor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size_bytes >= factor:
            return f"{size_bytes / factor:7.1f} {unit}"
    return f"{size_bytes:7d}   B"


def _fmt_time(dt) -> str:
    if dt is None:
        return " " * 19
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# sub-command handlers
# ---------------------------------------------------------------------------

def cmd_ls(args):
    key = _resolve_key_file(args)
    result = gcs_utils.list_gcs_prefix(
        gs_prefix=args.gs_prefix,
        key_file=key,
        recursive=args.recursive,
    )

    blobs    = result["blobs"]
    prefixes = result["prefixes"]
    bucket   = result["bucket"]
    prefix   = result["prefix"]

    base = f"gs://{bucket}/{prefix}/" if prefix else f"gs://{bucket}/"
    print(f"Listing: {base}{'  (recursive)' if args.recursive else ''}")

    if not blobs and not prefixes:
        print("  (empty)")
        return

    col_w = 12  # size column width
    print(f"\n{'SIZE':>{col_w}}  {'MODIFIED':<19}  PATH")
    print("-" * (col_w + 2 + 19 + 2 + 60))

    # Subdirectory prefixes first (non-recursive only)
    for p in prefixes:
        # strip gs://bucket/ and the listed prefix to get the relative name
        rel = p.removeprefix(f"gs://{bucket}/")
        if prefix:
            rel = rel.removeprefix(prefix).lstrip("/")
        print(f"{'DIR':>{col_w}}  {' ' * 19}  {rel}")

    # Blobs
    for entry in sorted(blobs, key=lambda b: b["name"]):
        rel = entry["name"]
        if prefix:
            rel = entry["name"].removeprefix(prefix).lstrip("/")
        print(f"{_fmt_size(entry['size']):>{col_w}}  {_fmt_time(entry['updated'])}  {rel}")

    # Summary line
    total = sum(e["size"] for e in blobs)
    n_dirs = len(prefixes)
    print("-" * (col_w + 2 + 19 + 2 + 60))
    parts = [f"{len(blobs)} file(s)"]
    if n_dirs:
        parts.append(f"{n_dirs} subdir(s)")
    parts.append(f"total {_fmt_size(total).strip()}")
    print("  ".join(parts))


def cmd_upload_file(args):
    key = _resolve_key_file(args)
    gcs_utils.upload_gcs_file(
        local_path=args.local_path,
        gs_uri=args.gs_uri,
        key_file=key,
        overwrite=args.overwrite,
        show_progress=True if args.progress else None,
    )


def cmd_upload_dir(args):
    key = _resolve_key_file(args)
    gcs_utils.upload_gcs_dir(
        local_dir=args.local_dir,
        gs_prefix=args.gs_prefix,
        key_file=key,
        overwrite=args.overwrite,
        pattern=args.pattern,
        show_progress=True if args.progress else None,
    )


def cmd_download_file(args):
    key = _resolve_key_file(args)
    gcs_utils.download_gcs_file(
        gs_uri=args.gs_uri,
        local_dir=args.local_dir,
        key_file=key,
        overwrite=args.overwrite,
        show_progress=True if args.progress else None,
    )


def cmd_download_dir(args):
    key = _resolve_key_file(args)
    gcs_utils.download_gcs_dir(
        gs_prefix=args.gs_prefix,
        local_dir=args.local_dir,
        key_file=key,
        overwrite=args.overwrite,
        pattern=args.pattern,
        show_progress=True if args.progress else None,
    )


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # Shared parent: key-file only (used by ls)
    parent_key = argparse.ArgumentParser(add_help=False)
    parent_key.add_argument(
        "--key-file", metavar="JSON",
        help="GCS service-account JSON key file "
             "(default: $GCS_KEY_FILE environment variable).",
    )

    # Shared parent: key-file + overwrite (used by transfer commands)
    parent_transfer = argparse.ArgumentParser(add_help=False)
    parent_transfer.add_argument(
        "--key-file", metavar="JSON",
        help="GCS service-account JSON key file "
             "(default: $GCS_KEY_FILE environment variable).",
    )
    parent_transfer.add_argument(
        "--overwrite", action="store_true",
        help="Re-upload/re-download even if a matching file already exists.",
    )
    parent_transfer.add_argument(
        "--progress", action="store_true",
        help="Always show a tqdm progress bar (default: auto — shown in a TTY, "
             "silent when output is redirected).",
    )

    parser = argparse.ArgumentParser(
        prog="gcs.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ls
    p = sub.add_parser(
        "ls", parents=[parent_key],
        help="List objects under a GCS prefix with size and modification time.",
    )
    p.add_argument("gs_prefix", metavar="GS_PREFIX",
                   help="GCS prefix to list, e.g. gs://bucket/runs/run01/")
    p.add_argument(
        "-r", "--recursive", action="store_true",
        help="List all objects recursively (default: top-level items only).",
    )
    p.set_defaults(func=cmd_ls)

    # upload-file
    p = sub.add_parser(
        "upload-file", parents=[parent_transfer],
        help="Upload a single local file to a GCS URI.",
    )
    p.add_argument("local_path", metavar="LOCAL_PATH",
                   help="Path to the local file to upload.")
    p.add_argument("gs_uri", metavar="GS_URI",
                   help="Destination GCS URI, e.g. gs://bucket/path/file.pt")
    p.set_defaults(func=cmd_upload_file)

    # upload-dir
    p = sub.add_parser(
        "upload-dir", parents=[parent_transfer],
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
        "download-file", parents=[parent_transfer],
        help="Download a single GCS object to a local directory.",
    )
    p.add_argument("gs_uri", metavar="GS_URI",
                   help="Source GCS URI, e.g. gs://bucket/path/file.root")
    p.add_argument("local_dir", metavar="LOCAL_DIR",
                   help="Local directory to write the downloaded file into.")
    p.set_defaults(func=cmd_download_file)

    # download-dir
    p = sub.add_parser(
        "download-dir", parents=[parent_transfer],
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
