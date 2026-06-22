# propella-annotation

Utilities for creating document-level annotation parquet files with
`ellamind/propella-1-4b` through an OpenAI-compatible local server such as vLLM.

This repository contains the annotation-related code extracted from the Danish
Dynaword workflow:

- `src/dynaword/propella.py`: prompt, taxonomy, and strict JSON schema.
- `src/dynaword/annotations.py`: annotation runner with local parquet output,
  resume support, and optional Hub upload.
- `src/dynaword/upload_annotations.py`: upload already-created local annotation
  parquet files.
- `src/tests/`: tests for parsing, retry behavior, parquet writing, resume, and
  upload selection.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Start Propella

Run an OpenAI-compatible server in one terminal:

```bash
vllm serve ellamind/propella-1-4b \
  --host 127.0.0.1 \
  --port 8000
```

The annotation client defaults to:

```text
model: ellamind/propella-1-4b
base URL: http://127.0.0.1:8000/v1
API key: EMPTY
```

## Create Annotations

Annotate one Danish Dynaword config:

```bash
python -m dynaword.annotations --dataset gutenberg
```

Annotate every non-default config from
`danish-foundation-models/danish-dynaword`:

```bash
python -m dynaword.annotations
```

Outputs are written as:

```text
annotations/<dataset>/ellamind--propella-1-4b.parquet
```

If a local dataset parquet exists under `data/<dataset>/<dataset>.parquet`, the
runner uses it. If the local parquet is missing or is only a Git LFS pointer, the
runner loads the dataset config from the Hugging Face Hub.

Interrupted runs can be resumed by running the same command again. Temporary
part files are counted and merged into the final parquet when all rows are done.

## Upload Existing Annotations

Set a token with write access:

```bash
export HF_TOKEN='<your-token>'
```

Upload one dataset annotation:

```bash
python -m dynaword.upload_annotations --dataset gutenberg
```

Upload all complete local annotation parquets:

```bash
python -m dynaword.upload_annotations
```

The default upload target is:

```text
danish-foundation-models/dynaword-annotations
```

## Test

```bash
pytest
```
