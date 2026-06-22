"""Create dataset annotations with a local vLLM server and upload them to the Hub."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi
from openai import OpenAI
from pydantic import ValidationError
from transformers import AutoTokenizer

from dynaword.annotation_config import DEFAULT_ANNOTATION_CONFIG
from dynaword.paths import annotations_path, data_path, repo_path
from dynaword.propella import (
    AnnotationResponse,
    create_messages,
    get_annotation_response_schema,
)

logger = logging.getLogger(__name__)

DATASET_NAMES: list[str] | None = None

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "AnnotationResponse",
        "schema": get_annotation_response_schema(),
        "strict": True,
    },
}


class TextAnnotator(Protocol):
    def annotate(self, text: str) -> dict[str, Any]: ...


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> list[int]: ...


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create document-level annotations for all datasets using Propella."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=DATASET_NAMES,
        help="Use to annotate a single dataset instead of all datasets.",
    )
    return parser


def get_dataset_names() -> list[str]:
    if DATASET_NAMES is not None:
        return DATASET_NAMES

    config_names = get_dataset_config_names(
        DEFAULT_ANNOTATION_CONFIG.dataset_repo_id,
    )
    return [config_name for config_name in config_names if config_name != "default"]


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def get_annotation_path(dataset_name: str, model_name: str) -> Path:
    return annotations_path / dataset_name / f"{model_slug(model_name)}.parquet"


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN before uploading annotations to the Hub.")
    return token


def get_upload_path(output_path: Path) -> str:
    return output_path.relative_to(repo_path).as_posix()


def get_temp_annotation_path(output_path: Path) -> Path:
    return output_path.with_suffix(".tmp.parquet")


def get_resume_annotation_path(output_path: Path, part_index: int) -> Path:
    return output_path.with_suffix(f".part{part_index}.parquet")


def parse_annotation(raw_annotation: str) -> dict[str, Any]:
    annotation = AnnotationResponse.model_validate_json(raw_annotation)
    return cast(dict[str, Any], annotation.model_dump(mode="json"))


def count_prompt_tokens(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def fits_context_window(prompt_tokens: int, max_output_tokens: int) -> bool:
    return (
        prompt_tokens
        + max_output_tokens
        + DEFAULT_ANNOTATION_CONFIG.prompt_token_margin
        <= DEFAULT_ANNOTATION_CONFIG.max_context_tokens
    )


def create_token_bounded_messages(
    text: str,
    tokenizer: ChatTokenizer,
    max_input_tokens: int,
) -> list[dict[str, str]]:
    char_limit = min(len(text), DEFAULT_ANNOTATION_CONFIG.max_content_chars)
    messages = create_messages(text, max_content_chars=char_limit)

    if count_prompt_tokens(tokenizer, messages) <= max_input_tokens:
        return messages

    low = 0
    high = char_limit
    best_messages = create_messages(text, max_content_chars=0)

    while low <= high:
        mid = (low + high) // 2
        candidate_messages = create_messages(text, max_content_chars=mid)

        if count_prompt_tokens(tokenizer, candidate_messages) <= max_input_tokens:
            best_messages = candidate_messages
            low = mid + 1
        else:
            high = mid - 1

    return best_messages


class PropellaAnnotator:
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model_name: str,
        client: Any | None = None,
        tokenizer: ChatTokenizer | None = None,
    ) -> None:
        self.client = client or OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.max_output_tokens = [
            DEFAULT_ANNOTATION_CONFIG.max_new_tokens,
            DEFAULT_ANNOTATION_CONFIG.retry_max_new_tokens,
        ]

    def get_max_input_tokens(self, max_output_tokens: int) -> int:
        return (
            DEFAULT_ANNOTATION_CONFIG.max_context_tokens
            - max_output_tokens
            - DEFAULT_ANNOTATION_CONFIG.prompt_token_margin
        )

    def annotate(self, text: str) -> dict[str, Any]:
        base_messages = create_token_bounded_messages(
            text,
            tokenizer=cast(ChatTokenizer, self.tokenizer),
            max_input_tokens=self.get_max_input_tokens(self.max_output_tokens[0]),
        )
        base_prompt_tokens = count_prompt_tokens(
            cast(ChatTokenizer, self.tokenizer),
            base_messages,
        )
        last_error: Exception | None = None
        raw_annotation = ""

        for attempt, max_output_tokens in enumerate(self.max_output_tokens, start=1):
            messages = base_messages
            if attempt > 1 and not fits_context_window(
                base_prompt_tokens,
                max_output_tokens,
            ):
                messages = create_token_bounded_messages(
                    text,
                    tokenizer=cast(ChatTokenizer, self.tokenizer),
                    max_input_tokens=self.get_max_input_tokens(max_output_tokens),
                )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format=RESPONSE_FORMAT,
                max_tokens=max_output_tokens,
            )
            raw_annotation = response.choices[0].message.content or ""

            if not raw_annotation:
                last_error = ValueError("The model returned an empty annotation response.")
            else:
                try:
                    return parse_annotation(raw_annotation)
                except ValidationError as error:
                    last_error = error

            if attempt < len(self.max_output_tokens):
                log_message = (
                    "Retrying annotation after invalid model response on attempt %s/%s."
                )
                if messages is not base_messages:
                    log_message = (
                        "Retrying annotation after invalid model response on attempt "
                        "%s/%s with a smaller input budget."
                    )
                logger.warning(
                    log_message,
                    attempt,
                    len(self.max_output_tokens),
                )

        logger.error("Failed to parse model response:")
        logger.error("----- MODEL RESPONSE START -----")
        logger.error(raw_annotation)
        logger.error("----- MODEL RESPONSE END -----")
        assert last_error is not None
        raise last_error


def upload_annotation(api: HfApi, dataset_name: str, output_path: Path) -> None:
    path_in_repo = get_upload_path(output_path)
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo=path_in_repo,
        repo_id=DEFAULT_ANNOTATION_CONFIG.upload_repo_id,
        repo_type="dataset",
        commit_message=(
            f"Add {dataset_name} annotations with "
            f"{DEFAULT_ANNOTATION_CONFIG.model_name}"
        ),
    )
    logger.info(
        "Uploaded %s to %s/%s",
        output_path.name,
        DEFAULT_ANNOTATION_CONFIG.upload_repo_id,
        path_in_repo,
    )


def get_remote_annotation_paths(api: HfApi) -> set[str]:
    return set(
        api.list_repo_files(
            repo_id=DEFAULT_ANNOTATION_CONFIG.upload_repo_id,
            repo_type="dataset",
        )
    )


def get_partial_annotation_paths(output_path: Path) -> list[Path]:
    partial_paths: list[Path] = []
    temp_path = get_temp_annotation_path(output_path)
    if temp_path.exists():
        partial_paths.append(temp_path)

    part_index = 1
    while True:
        part_path = get_resume_annotation_path(output_path, part_index)
        if not part_path.exists():
            break
        partial_paths.append(part_path)
        part_index += 1

    return partial_paths


def count_parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def merge_annotation_parts(partial_paths: list[Path], output_path: Path) -> None:
    schema = pq.ParquetFile(partial_paths[0]).schema_arrow
    merged_path = output_path.with_suffix(".merged.parquet")
    if merged_path.exists():
        merged_path.unlink()

    with pq.ParquetWriter(merged_path, schema) as writer:
        for partial_path in partial_paths:
            parquet_file = pq.ParquetFile(partial_path)
            for row_group_index in range(parquet_file.num_row_groups):
                writer.write_table(parquet_file.read_row_group(row_group_index))

    merged_path.replace(output_path)
    for partial_path in partial_paths:
        partial_path.unlink()


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False

    with path.open("rb") as handle:
        return handle.readline().strip() == b"version https://git-lfs.github.com/spec/v1"


def load_annotation_dataset(dataset_name: str, dataset_path: Path) -> Dataset:
    if dataset_path.exists() and not is_git_lfs_pointer(dataset_path):
        dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
        return cast(Dataset, dataset)

    logger.info(
        "Loading dataset '%s' from %s on the Hub because %s is not a materialized parquet file.",
        dataset_name,
        DEFAULT_ANNOTATION_CONFIG.dataset_repo_id,
        dataset_path,
    )
    dataset = load_dataset(
        DEFAULT_ANNOTATION_CONFIG.dataset_repo_id,
        dataset_name,
        split="train",
    )
    return cast(Dataset, dataset)


def create_record(sample: dict[str, Any], annotator: TextAnnotator) -> dict[str, Any]:
    return {"id": sample["id"], **annotator.annotate(sample["text"])}


def create_records(
    samples: list[dict[str, Any]],
    annotator: TextAnnotator,
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    annotations = executor.map(annotator.annotate, [sample["text"] for sample in samples])
    return [
        {"id": sample["id"], **annotation}
        for sample, annotation in zip(samples, annotations, strict=True)
    ]


def annotate_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_path: Path,
    annotator: TextAnnotator,
) -> None:
    dataset = load_annotation_dataset(dataset_name, dataset_path)

    if len(dataset) == 0:
        logger.warning("Dataset '%s' contains no rows, skipping.", dataset_name)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_paths = get_partial_annotation_paths(output_path)
    completed_rows = sum(count_parquet_rows(path) for path in partial_paths)

    if completed_rows > len(dataset):
        raise ValueError(
            f"Found {completed_rows} partial annotation rows for '{dataset_name}', "
            f"but the dataset only contains {len(dataset)} rows."
        )

    if completed_rows == len(dataset):
        logger.info(
            "Finalizing existing partial annotations for '%s' (%s rows).",
            dataset_name,
            completed_rows,
        )
        merge_annotation_parts(partial_paths, output_path)
        return

    active_output_path = (
        get_temp_annotation_path(output_path)
        if not partial_paths
        else get_resume_annotation_path(output_path, len(partial_paths))
    )
    if active_output_path.exists():
        active_output_path.unlink()

    concurrency = min(DEFAULT_ANNOTATION_CONFIG.annotation_concurrency, len(dataset))
    remaining_rows = len(dataset) - completed_rows
    batch_size = min(DEFAULT_ANNOTATION_CONFIG.annotation_batch_size, remaining_rows)
    batch_size = max(batch_size, concurrency)
    batch_stop = min(completed_rows + batch_size, len(dataset))
    first_batch = [
        cast(dict[str, Any], dataset[index])
        for index in range(completed_rows, batch_stop)
    ]

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        first_records = create_records(first_batch, annotator=annotator, executor=executor)
        first_table = pa.Table.from_pylist(first_records)

        with pq.ParquetWriter(active_output_path, first_table.schema) as writer:
            writer.write_table(first_table)
            logger.info(
                "Annotated %s/%s rows for %s",
                completed_rows + len(first_records),
                len(dataset),
                dataset_name,
            )

            for start in range(batch_stop, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                batch = [cast(dict[str, Any], dataset[index]) for index in range(start, stop)]
                records = create_records(batch, annotator=annotator, executor=executor)
                writer.write_table(pa.Table.from_pylist(records))

                if stop == len(dataset) or stop % 25 == 0:
                    logger.info(
                        "Annotated %s/%s rows for %s",
                        stop,
                        len(dataset),
                        dataset_name,
                    )

    partial_paths.append(active_output_path)
    if len(partial_paths) == 1:
        active_output_path.replace(output_path)
        return

    merge_annotation_parts(partial_paths, output_path)


def main(dataset: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    annotator = PropellaAnnotator(
        api_base_url=DEFAULT_ANNOTATION_CONFIG.api_base_url,
        api_key=DEFAULT_ANNOTATION_CONFIG.api_key,
        model_name=DEFAULT_ANNOTATION_CONFIG.model_name,
    )
    hub_api = HfApi(token=get_hf_token())
    hub_api.create_repo(
        repo_id=DEFAULT_ANNOTATION_CONFIG.upload_repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    remote_annotation_paths = get_remote_annotation_paths(hub_api)

    datasets = get_dataset_names() if dataset is None else [dataset]
    for dataset_name in datasets:
        dataset_path = data_path / dataset_name / f"{dataset_name}.parquet"
        output_path = get_annotation_path(
            dataset_name, DEFAULT_ANNOTATION_CONFIG.model_name
        )
        upload_path = get_upload_path(output_path)

        if upload_path in remote_annotation_paths:
            logger.info(
                "Skipping dataset '%s' because %s already exists in %s.",
                dataset_name,
                upload_path,
                DEFAULT_ANNOTATION_CONFIG.upload_repo_id,
            )
            continue

        if output_path.exists():
            output_path.unlink()

        logger.info(
            "Annotating dataset '%s' with %s via %s",
            dataset_name,
            DEFAULT_ANNOTATION_CONFIG.model_name,
            DEFAULT_ANNOTATION_CONFIG.api_base_url,
        )
        annotate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            output_path=output_path,
            annotator=annotator,
        )
        if output_path.exists():
            upload_annotation(hub_api, dataset_name=dataset_name, output_path=output_path)
            remote_annotation_paths.add(upload_path)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(dataset=args.dataset)
