"""Upload existing local annotation parquet files to the Hub."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from dynaword.annotation_config import DEFAULT_ANNOTATION_CONFIG
from dynaword.annotations import (
    DATASET_NAMES,
    get_dataset_names,
    get_annotation_path,
    get_hf_token,
    load_annotation_dataset,
    upload_annotation,
)
from dynaword.paths import data_path

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload existing local annotation parquet files to the Hub."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=DATASET_NAMES,
        help="Use to upload a single dataset annotation instead of all local outputs.",
    )
    return parser


def get_local_annotation_outputs(dataset: str | None = None) -> list[tuple[str, Path]]:
    datasets = (
        DATASET_NAMES
        if dataset is None and DATASET_NAMES is not None
        else get_dataset_names()
        if dataset is None
        else [dataset]
    )
    outputs: list[tuple[str, Path]] = []

    for dataset_name in datasets:
        output_path = get_uploadable_annotation_output(dataset_name)
        if output_path is not None:
            outputs.append((dataset_name, output_path))

    return outputs


def get_uploadable_annotation_output(dataset_name: str) -> Path | None:
    output_path = get_annotation_path(
        dataset_name,
        DEFAULT_ANNOTATION_CONFIG.model_name,
    )
    if output_path.exists():
        return output_path

    temp_path = output_path.with_suffix(".tmp.parquet")
    if not temp_path.exists():
        return None

    dataset_path = data_path / dataset_name / f"{dataset_name}.parquet"
    dataset = load_annotation_dataset(dataset_name, dataset_path)
    expected_rows = len(dataset)
    actual_rows = pq.ParquetFile(temp_path).metadata.num_rows

    if actual_rows == expected_rows:
        logger.info(
            "Using completed temporary annotation parquet for '%s' (%s rows).",
            dataset_name,
            actual_rows,
        )
        return temp_path

    logger.warning(
        "Skipping temporary annotation parquet for '%s' because it has %s/%s rows.",
        dataset_name,
        actual_rows,
        expected_rows,
    )
    return None


def main(dataset: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    outputs = get_local_annotation_outputs(dataset)
    if not outputs:
        logger.warning("No local annotation parquet files were found to upload.")
        return

    hub_api = HfApi(token=get_hf_token())
    hub_api.create_repo(
        repo_id=DEFAULT_ANNOTATION_CONFIG.upload_repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    for dataset_name, output_path in outputs:
        upload_annotation(hub_api, dataset_name=dataset_name, output_path=output_path)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(dataset=args.dataset)
