# FNO4d Training — Google Cloud Storage

Training on ROOT files stored in Google Cloud Storage without requiring a
local copy of the dataset on the training machine.

> **Base documentation** — this file describes only the GCS-specific steps.
> For parameters, data format, normalisation, model architecture, metrics,
> checkpoints, and output files see [README_train.md](README_train.md).

> **Setup and environment** — see the main [README.md](README.md) for
> installation, environment activation, pre-flight checks, and GCS package
> installation (`--gcs` flag).

---

## Two access modes

| Mode | Script suffix | How data is read | Best for |
|------|--------------|-----------------|---------|
| **Streaming** | `_gcs_stream` | `uproot` reads directly from `gs://` via `fsspec`/`gcsfs` — no data lands on disk | Low local-disk instances; repeated one-off runs on different bucket files |
| **Download** | `_gcs_download` | Files are copied to `local_data_dir` first, then read like local files | Repeated runs on the same files; shared network filesystems where streaming is slow; GPU nodes with large local SSD |

Both modes authenticate with the same service-account JSON key file and
produce identical model artifacts.

---

## Prerequisites

### 1 — GCS packages

Install the GCS optional extras (not included in the base install):

```bash
bash install.sh --gcs
# or, in an already-active env:
pip install -r requirements-gcs.txt
# or via the loc_libs extras:
pip install -e './loc_libs[gcs]'
```

This installs `gcsfs` (fsspec driver), `google-cloud-storage` (download API),
and `tqdm` (byte-level progress bar for downloads).
Both `gcsfs` and `google-cloud-storage` are required for download mode; only
`gcsfs` is needed for streaming, but installing all three together keeps the
two modes interchangeable.

### 2 — Service-account key file

GCS authentication uses a **service-account JSON key file**.

1. In the Google Cloud Console, navigate to
   **IAM & Admin → Service Accounts → Create service account**.
2. Grant the account the **Storage Object Viewer** role.  There are two ways:
   - **Project-wide** (IAM & Admin → IAM): the account can read from all
     buckets in the project.
   - **Bucket-level** (Cloud Storage → Buckets → *bucket* → Permissions →
     Grant access): the account is restricted to that specific bucket.
     In many organisation setups the service account does **not** receive
     project-wide roles automatically — you must add it explicitly to each
     bucket it needs to access.
3. Under **Keys → Add key → Create new key**, select **JSON** and download the
   file.  Keep it outside the repo and never commit it to git.

```
/path/to/service_account.json   # example location — NOT inside the repo
```

> **Minimum IAM permissions required**
>
> | Operation | Role |
> |-----------|------|
> | Streaming (read-only) | `roles/storage.objectViewer` |
> | Download (read-only) | `roles/storage.objectViewer` |
> | List bucket contents (optional) | `roles/storage.objectViewer` (includes `storage.objects.list`) |
>
> **Project-wide vs bucket-level access** — In some Google Cloud organisations
> the service account is granted roles only on specific buckets, not
> project-wide.  If access to a bucket is unexpectedly denied, check the
> bucket's own **Permissions** tab (Cloud Storage → Buckets → *bucket* →
> Permissions) and add the service account there with the required role even
> if it already appears in the project-level IAM policy.

---

## Scripts

Four scripts live in `train_gcs/`:

| Script | Backend | Mode |
|--------|---------|------|
| `test_train_yaml_norm_gcs_stream.py` | PyTorch | GCS streaming |
| `test_train_mlx_yaml_norm_gcs_stream.py` | MLX | GCS streaming |
| `test_train_yaml_norm_gcs_download.py` | PyTorch | GCS download |
| `test_train_mlx_yaml_norm_gcs_download.py` | MLX | GCS download |

Each script is a direct equivalent of its `train/test_train_yaml_norm.py` or
`train/test_train_mlx_yaml_norm.py` counterpart and accepts the same `--config`
flag:

```bash
cd train_gcs/

# PyTorch streaming
python test_train_yaml_norm_gcs_stream.py --config config_gcs_stream.yaml

# MLX streaming (Apple Silicon)
python test_train_mlx_yaml_norm_gcs_stream.py --config config_gcs_stream.yaml

# PyTorch download
python test_train_yaml_norm_gcs_download.py --config config_gcs_download.yaml

# MLX download
python test_train_mlx_yaml_norm_gcs_download.py --config config_gcs_download.yaml
```

---

## YAML configuration

### Streaming — `config_gcs_stream.yaml`

The config is identical to `train/config_test_train.yaml` except for two
differences:

1. `gcs_key_file` replaces `input_dir` as the GCS credential pointer.
2. `input_files` must be **full `gs://` URIs** (not bare filenames).
   `input_dir` is ignored by the streaming scripts.

```yaml
# --- GCS (new keys, required) ---
gcs_key_file: /path/to/service_account.json

# --- input data ---
input_files:
  - gs://my-bucket/data/dAu_25ev_mb.root

# --- everything else is unchanged from config_test_train.yaml ---
odir: ./out
model_dir: ./checkpoints
nevents_in: 10
normalization:
  name: none
train_ratio: 0.8
xyslice:  [null, -1]
tauslice: [null, -1]
hidden_channels: 16
n_modes: [32, 32, 16, 16]
n_epochs: 3
batch_size: 1
lr: 8.0e-3
weight_decay: 1.0e-4
```

### Download — `config_gcs_download.yaml`

Same as the streaming config plus three additional keys that control the local
cache:

```yaml
# --- GCS (new keys, required) ---
gcs_key_file: /path/to/service_account.json

# --- local cache (new keys for download mode) ---
local_data_dir: ./data      # directory where .root files are stored locally
overwrite_cache: false      # true = always re-download even if the file exists
gcs_show_progress: null     # null = auto (bar in terminal, silent in batch/redirect)
                            # true  = always show bar | false = never show bar

# --- input data ---
input_files:
  - gs://my-bucket/data/dAu_25ev_mb.root

# --- everything else unchanged ---
```

#### Download caching behaviour

| Condition | Result |
|-----------|--------|
| Local file absent | Downloaded from GCS |
| Local file present, same size as remote blob | Skipped — cached file is used |
| Local file present, size differs | Re-downloaded (file is corrupted or stale) |
| `overwrite_cache: true` | Always re-downloaded regardless of local state |

The size comparison uses the blob's `Content-Length` metadata, which GCS
returns without an extra download.  For large ROOT files (multi-GiB) the
cached path typically saves minutes per run on subsequent training iterations.

---

## New YAML keys at a glance

| Key | Used by | Required | Default | Description |
|-----|---------|----------|---------|-------------|
| `gcs_key_file` | both | yes | — | Path to service-account JSON key file |
| `gcs_output_prefix` | both | no | `null` | Upload `model_dir` and `odir` here after training (e.g. `gs://my-bucket/runs/run01`) |
| `local_data_dir` | download only | no | `./data` | Local directory for downloaded files |
| `overwrite_cache` | download only | no | `false` | Force re-download even if cached file exists |
| `gcs_show_progress` | download only | no | `null` | Progress bar: `null` = auto (shown in terminal / Jupyter, silent in batch); `true` = always; `false` = never |

`input_dir` is **not** used by either GCS mode — `input_files` entries are
full `gs://` URIs and are passed directly to `read_3d_data` (streaming) or
`download_gcs_files` (download).

---

## Syncing model artifacts between GCS and local disk

`upload_gcs_dir` and `download_gcs_dir` are symmetric counterparts — same
arguments, same `pattern` filter, same size-based skip logic, opposite
direction.  Both preserve the relative directory structure.

```
local_dir/best_model.safetensors    ↔    gs_prefix/best_model.safetensors
local_dir/checkpoint_10/model.pt    ↔    gs_prefix/checkpoint_10/model.pt
```

---

## Uploading checkpoints and model artifacts to GCS

After training, call `upload_gcs_dir` to push the entire `model_dir` (and/or
`odir`) back to GCS.  The size-check skips anything already uploaded, so
re-running the call after adding new checkpoints only transfers the new files.

### Upload a full output directory

```python
import sys
sys.path.insert(0, 'loc_libs')
from gcs_utils import upload_gcs_dir

# push everything in model_dir to GCS
gs_uris = upload_gcs_dir(
    local_dir  = './checkpoints',
    gs_prefix  = 'gs://my-bucket/runs/run01/checkpoints',
    key_file   = '/path/to/key.json',
    overwrite  = False,   # skip blobs that already exist with the same size
)
# gs_uris is a list of every gs:// URI written or already present
```

### Upload specific file types only

The `pattern` argument is a standard `pathlib` glob relative to `local_dir`:

```python
# safetensors and JSON files only (model weights + metadata)
upload_gcs_dir('./checkpoints', 'gs://my-bucket/runs/run01/checkpoints',
               '/path/to/key.json', pattern='**/*.safetensors')

upload_gcs_dir('./checkpoints', 'gs://my-bucket/runs/run01/checkpoints',
               '/path/to/key.json', pattern='**/*.json')

# periodic checkpoint subdirs only
upload_gcs_dir('./checkpoints', 'gs://my-bucket/runs/run01/checkpoints',
               '/path/to/key.json', pattern='checkpoint_*/**/*')
```

### Upload a single file

```python
from gcs_utils import upload_gcs_file

upload_gcs_file(
    local_path = './checkpoints/best_model.safetensors',
    gs_uri     = 'gs://my-bucket/runs/run01/checkpoints/best_model.safetensors',
    key_file   = '/path/to/key.json',
)
```

### Post-training upload — built into the training scripts

All four `train_gcs/` scripts include this block after the final timing log:

```python
if gcs_output_prefix:
    log(f'Uploading outputs to GCS: {gcs_output_prefix}')
    if log._f:
        log._f.flush()   # ensure the log file is complete before uploading
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file)
    upload_gcs_dir(odir,      f'{gcs_output_prefix}/logs',        gcs_key_file)
    log(f'GCS upload complete → {gcs_output_prefix}')
```

To activate it, uncomment and set `gcs_output_prefix` in the YAML config:

```yaml
gcs_output_prefix: gs://my-bucket/runs/run01
```

What gets uploaded:

| Local path | GCS destination |
|-----------|----------------|
| `model_dir/` | `{gcs_output_prefix}/checkpoints/` |
| `odir/` | `{gcs_output_prefix}/logs/` |

`model_dir` holds the model weights, optimizer state, periodic checkpoints,
and `normalizer.npz`.  `odir` holds the log file, `metrics*.json`, and
`training_curves*.svg`.  The log file is explicitly flushed before the upload
so the full training log — including the timing line — is captured.

The upload uses `overwrite=False` by default, so re-running a script that
crashed mid-training only transfers the files that changed since the last
successful upload.

### Required IAM permissions for upload

The service account used for uploading needs write access in addition to
read:

| Operation | Minimum role |
|-----------|-------------|
| Upload (write) | `roles/storage.objectCreator` |
| Upload + overwrite | `roles/storage.objectUser` (= Creator + delete) |
| Full read + write | `roles/storage.objectAdmin` |

These roles must be granted on the **destination bucket** — either via the
project-level IAM policy (IAM & Admin → IAM) or directly on the bucket
(Cloud Storage → Buckets → *bucket* → Permissions → Grant access).  In
organisations where service accounts do not receive project-wide roles
automatically, the bucket-level grant is required even if the account can
read from other buckets.

A separate upload-only key can be used alongside the read-only download key
if the training machine and the storage destination are in different trust
domains.

---

## Downloading a GCS directory to local disk

### Option A — `gsutil` (simplest, no Python needed)

```bash
# install the Cloud SDK once if needed
pip install gsutil
# or via the full SDK: https://cloud.google.com/sdk/docs/install

# download an entire GCS prefix to a local folder
gsutil -m cp -r gs://my-bucket/data/ ./data/

# with explicit credentials (service-account key)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json gsutil -m cp -r gs://my-bucket/data/ ./data/
```

`-m` enables parallel transfers; `-r` is recursive.

---

### Option B — `download_gcs_dir` (Python, symmetric with `upload_gcs_dir`)

The primary Python option for downloading a directory.  It mirrors
`upload_gcs_dir` exactly: same argument names, same `pattern` filter,
same size-based skip logic, opposite direction.

```python
import sys
sys.path.insert(0, 'loc_libs')
from gcs_utils import download_gcs_dir

# download everything
local_paths = download_gcs_dir(
    gs_prefix = 'gs://my-bucket/runs/run01/checkpoints',
    local_dir = './checkpoints',
    key_file  = '/path/to/key.json',
    overwrite = False,    # skip files already cached with matching size
    pattern   = '**/*',  # default — download everything
)

# download only model weights and metadata
local_paths = download_gcs_dir(
    'gs://my-bucket/runs/run01/checkpoints', './checkpoints',
    '/path/to/key.json', pattern='**/*.safetensors')

local_paths = download_gcs_dir(
    'gs://my-bucket/runs/run01/checkpoints', './checkpoints',
    '/path/to/key.json', pattern='**/*.json')

# download only a specific checkpoint subdirectory
local_paths = download_gcs_dir(
    'gs://my-bucket/runs/run01/checkpoints', './checkpoints',
    '/path/to/key.json', pattern='checkpoint_50/**/*')
```

The prefix-relative directory structure is always preserved:

```
gs_prefix/best_model.safetensors    →  local_dir/best_model.safetensors
gs_prefix/checkpoint_10/model.pt    →  local_dir/checkpoint_10/model.pt
gs_prefix/last_state.json           →  local_dir/last_state.json
```

> **Pattern note** — the `**/<tail>` form (e.g. `**/*.safetensors`) works on
> all Python versions ≥ 3.9.  Arbitrary `**` placement (e.g.
> `checkpoint_*/**/*`) requires Python 3.12+ for full correctness; on earlier
> versions use `download_gcs_prefix` with explicit blob enumeration instead.

### Option C — `download_gcs_prefix` (Python, flat or structured, no pattern filter)

Use when you need `preserve_structure=False` (all files land flat in
`local_dir`) or when working with data files that don't need pattern
filtering:

```python
from gcs_utils import download_gcs_prefix

local_paths = download_gcs_prefix(
    gs_prefix          = 'gs://my-bucket/data/',
    local_dir          = './data',
    key_file           = '/path/to/key.json',
    overwrite          = False,
    preserve_structure = True,   # False = flat (basename only — beware collisions)
)
```

After the sync, pass the returned `local_paths` directly to the training
scripts, or point `local_data_dir` in the YAML to the same folder and list the
downloaded filenames in `input_files` as `gs://` URIs (the download scripts
will find them already cached).

---

## `loc_libs/gcs_utils` API

The GCS helpers live in `loc_libs/gcs_utils.py` and are imported by the
training scripts.  They can also be called directly for scripting or
interactive use.

```python
import sys
sys.path.insert(0, 'loc_libs')
from gcs_utils import (
    get_gcs_storage_options,
    download_gcs_files, download_gcs_prefix,
    download_gcs_dir,
    upload_gcs_file,   upload_gcs_dir,
)

# --- streaming: get storage_options for read_3d_data ---
opts = get_gcs_storage_options('/path/to/key.json')
from read_3d_root import read_3d_data
data = read_3d_data('gs://my-bucket/data/file.root', storage_options=opts)

# --- download specific files ---
local_paths = download_gcs_files(
    gs_uris       = ['gs://my-bucket/data/file1.root'],
    local_dir     = './data',
    key_file      = '/path/to/key.json',
    overwrite     = False,
    show_progress = None,   # None = auto (terminal/Jupyter); True/False = force on/off
)

# --- download a directory (with optional pattern filter) ---
local_paths = download_gcs_dir(
    gs_prefix     = 'gs://my-bucket/runs/run01/checkpoints',
    local_dir     = './checkpoints',
    key_file      = '/path/to/key.json',
    overwrite     = False,
    pattern       = '**/*',   # glob relative to gs_prefix; default = everything
    show_progress = None,
)

# --- download a directory (flat layout, no pattern filter) ---
local_paths = download_gcs_prefix(
    gs_prefix          = 'gs://my-bucket/data/',
    local_dir          = './data',
    key_file           = '/path/to/key.json',
    overwrite          = False,
    preserve_structure = True,   # False = flat (basename only)
    show_progress      = None,
)

# --- upload a single file ---
upload_gcs_file('./checkpoints/best_model.safetensors',
                'gs://my-bucket/runs/run01/checkpoints/best_model.safetensors',
                '/path/to/key.json')

# --- upload a whole directory (checkpoints, logs, …) ---
upload_gcs_dir(
    local_dir = './checkpoints',
    gs_prefix = 'gs://my-bucket/runs/run01/checkpoints',
    key_file  = '/path/to/key.json',
    overwrite = False,
    pattern   = '**/*',          # glob relative to local_dir; default = everything
)
```

The module uses lazy import guards: importing `gcs_utils` succeeds even when
`gcsfs` / `google-cloud-storage` are absent.  An `ImportError` with an
install hint is raised only when a GCS function is actually called.

```python
from gcs_utils import check_gcs_dependencies

check_gcs_dependencies()                  # check gcsfs only (streaming)
check_gcs_dependencies(require_storage=True)  # also check google-cloud-storage (download)
```

---

## Running on a remote GPU instance (Lambda / HPC)

### Streaming (no data transfer)

```bash
# 1 — install GCS packages in the instance env
bash install.sh --gcs

# 2 — copy the key file (never commit it to git)
scp /local/key.json user@instance:/secure/key.json

# 3 — set gcs_key_file in the config, then run
cd train_gcs/
python test_train_yaml_norm_gcs_stream.py --config config_gcs_stream.yaml
```

The ROOT files stay in GCS and are read via the network.  On a Lambda instance
with a 10 Gbps uplink the streaming overhead for a typical training run is
small compared to GPU compute time.  For very large files or many epochs,
consider the download mode instead.

### Download (cache locally, then train)

```bash
# 1–2 — same as streaming

# 3 — set gcs_key_file, local_data_dir, and input_files in the config
cd train_gcs/
python test_train_yaml_norm_gcs_download.py --config config_gcs_download.yaml
```

On the first run the files are downloaded to `local_data_dir`.  All subsequent
runs — including resuming after a checkpoint — reuse the cached local files.
This is the recommended mode when training for many epochs on a persistent
instance.

---

## Normalisation and inference

The `normalizer.npz` file saved to `model_dir` is identical to the one
produced by the local `_norm` scripts and can be loaded the same way at
inference time:

```python
from norm_utils import Normalizer

normalizer = Normalizer.load('checkpoints/normalizer.npz')
pred_physical = normalizer.inverse(model_output_array)
```

For the streaming mode, inference-time data can also be read directly from
GCS using the same `storage_options` pattern:

```python
from gcs_utils import get_gcs_storage_options
from read_3d_root import read_3d_data

opts = get_gcs_storage_options('/path/to/key.json')
data = read_3d_data('gs://my-bucket/data/test_file.root', storage_options=opts)
```

---

## Output files

The GCS scripts produce the same output files as their local equivalents — no
new files are created.  See [README_train.md — Output files](README_train.md#output-files)
for the full table.

The only additional file written by the download mode is the cached `.root`
data in `local_data_dir/`.  This directory is **not** touched by the
streaming mode.

---

## Troubleshooting

### `ImportError: gcsfs is required for GCS streaming`

`gcsfs` is not installed.  Run:

```bash
pip install -r requirements-gcs.txt
# or:
pip install -e './loc_libs[gcs]'
```

### `google.auth.exceptions.DefaultCredentialsError`

The key file path in `gcs_key_file` is wrong or the file does not exist.
Check with:

```bash
python -c "from pathlib import Path; print(Path('/your/key.json').exists())"
```

### `google.api_core.exceptions.Forbidden` / 403

The service account lacks permission on the bucket.  Verify it has at least
the **Storage Object Viewer** (`roles/storage.objectViewer`) role for reads,
or **Storage Object Creator / User** for uploads.

Check **both** permission levels — project-wide and bucket-level:
- Project-wide: GCP Console → IAM & Admin → IAM → find the service-account
  email and confirm the role appears there.
- Bucket-level: GCP Console → Cloud Storage → Buckets → *bucket* →
  Permissions → confirm the service-account email is listed.

In many organisation setups a service account does **not** automatically
inherit project-wide roles.  If the account is missing from the bucket's own
Permissions tab, add it there explicitly with the required role — a
project-level entry alone is not enough in those configurations.

### `FileNotFoundError: gs://…`

The blob path does not exist in the bucket.  List available files:

```python
from google.cloud import storage
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file('/path/to/key.json')
client = storage.Client(credentials=creds, project=creds.project_id)
for blob in client.list_blobs('my-bucket', prefix='data/'):
    print(blob.name, blob.size)
```

### Slow streaming on a HPC node

Network throughput between the HPC cluster and GCS may be limited.  Switch to
the download mode so the transfer happens once at the start of the job and
all subsequent reads (including DataLoader workers) hit local disk.
