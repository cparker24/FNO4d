# cloud_util

Command-line utilities for transferring files between local storage and Google Cloud Storage (GCS).

## Prerequisites

Install the GCS dependencies from the repo root:

```bash
pip install -r requirements-gcs.txt
# or:
bash install.sh --gcs
```

## Authentication

All commands require a GCS service-account JSON key file. Pass it via:

- `--key-file <path>` flag, **or**
- `GCS_KEY_FILE` environment variable:

```bash
export GCS_KEY_FILE=/path/to/service_account.json
```

## Usage

```
python gcs.py <command> [options] <positional args>
```

### Commands

| Command | Description |
|---|---|
| `ls` | List objects under a GCS prefix with size and modification time |
| `upload-file` | Upload a single local file to a GCS URI |
| `upload-dir` | Upload a local directory tree to a GCS prefix |
| `download-file` | Download a single GCS object to a local directory |
| `download-dir` | Download a GCS prefix (directory) to a local directory |

### Common flags

| Flag | Commands | Default | Description |
|---|---|---|---|
| `--key-file JSON` | all | `$GCS_KEY_FILE` | Service-account JSON key file |
| `-r`, `--recursive` | `ls` | off | List all objects recursively instead of top-level only |
| `--overwrite` | transfer commands | off | Re-transfer even if a size-matched copy already exists |
| `--pattern GLOB` | `*-dir` commands | `**/*` | Glob filter relative to the source root |

## Examples

**List top-level items** (files + subdirectories)
```bash
python gcs.py ls gs://my-bucket/runs/run01/ --key-file service_account.json
```

Output:
```
Listing: gs://my-bucket/runs/run01/

        SIZE  MODIFIED             PATH
------------------------------------------------------------
         DIR                       checkpoints/
         DIR                       logs/
   4.2 MiB  2025-03-15 14:22:01  model.pt
   1.1 KiB  2025-03-15 14:22:01  config.yaml
------------------------------------------------------------
2 file(s)  2 subdir(s)  total 4.2 MiB
```

**List all objects recursively**
```bash
python gcs.py ls gs://my-bucket/runs/run01/ -r --key-file service_account.json
```

**Upload a single file**
```bash
python gcs.py upload-file model.pt gs://my-bucket/runs/run01/model.pt \
    --key-file service_account.json
```

**Upload a directory** (everything)
```bash
python gcs.py upload-dir ./out/run01 gs://my-bucket/runs/run01/ \
    --key-file service_account.json
```

**Upload a directory** (safetensors files only)
```bash
python gcs.py upload-dir ./out/run01 gs://my-bucket/runs/run01/ \
    --key-file service_account.json --pattern '**/*.safetensors'
```

**Download a single file** into `./data/`
```bash
python gcs.py download-file gs://my-bucket/data/events.root ./data/ \
    --key-file service_account.json
```

**Download a full directory**
```bash
python gcs.py download-dir gs://my-bucket/runs/run01/ ./local/run01/ \
    --key-file service_account.json
```

**Download JSON files only**
```bash
python gcs.py download-dir gs://my-bucket/runs/run01/ ./local/run01/ \
    --key-file service_account.json --pattern '**/*.json'
```

**Force re-download** (ignore local cache)
```bash
python gcs.py download-dir gs://my-bucket/runs/run01/ ./local/run01/ \
    --key-file service_account.json --overwrite
```

## Caching behaviour

By default, transfers are skipped when a local file and the remote blob have the
same byte size. Use `--overwrite` to bypass this check and always transfer.

## Implementation

`gcs.py` is a thin CLI wrapper around `loc_libs/gcs_utils.py`, which provides
the underlying upload and download functions used throughout the FNO4d training
scripts.
