from pathlib import Path

from datasets import Dataset

from dynaword.upload_annotations import (
    get_local_annotation_outputs,
    get_uploadable_annotation_output,
)


def test_get_local_annotation_outputs(monkeypatch, tmp_path):
    existing_path = tmp_path / "enevaeldens_nyheder.parquet"
    existing_path.write_text("ok")

    def fake_get_annotation_path(dataset_name: str, model_name: str) -> Path:
        assert model_name == "ellamind/propella-1-4b"
        if dataset_name == "enevaeldens_nyheder":
            return existing_path
        return tmp_path / dataset_name / "missing.parquet"

    monkeypatch.setattr(
        "dynaword.upload_annotations.get_annotation_path",
        fake_get_annotation_path,
    )
    monkeypatch.setattr(
        "dynaword.upload_annotations.DATASET_NAMES",
        ["enevaeldens_nyheder", "gutenberg"],
    )

    assert get_local_annotation_outputs() == [("enevaeldens_nyheder", existing_path)]
    assert get_local_annotation_outputs("enevaeldens_nyheder") == [
        ("enevaeldens_nyheder", existing_path)
    ]


def test_get_uploadable_annotation_output_uses_complete_temp_file(
    monkeypatch, tmp_path
):
    temp_path = tmp_path / "ellamind--propella-1-4b.tmp.parquet"
    Dataset.from_dict({"id": ["row-1", "row-2"], "text": ["a", "b"]}).to_parquet(
        str(temp_path)
    )

    def fake_get_annotation_path(dataset_name: str, model_name: str) -> Path:
        assert dataset_name == "enevaeldens_nyheder"
        assert model_name == "ellamind/propella-1-4b"
        return tmp_path / "ellamind--propella-1-4b.parquet"

    def fake_load_annotation_dataset(dataset_name: str, dataset_path: Path):
        assert dataset_name == "enevaeldens_nyheder"
        return Dataset.from_dict({"id": ["row-1", "row-2"], "text": ["a", "b"]})

    monkeypatch.setattr(
        "dynaword.upload_annotations.get_annotation_path",
        fake_get_annotation_path,
    )
    monkeypatch.setattr(
        "dynaword.upload_annotations.load_annotation_dataset",
        fake_load_annotation_dataset,
    )

    assert get_uploadable_annotation_output("enevaeldens_nyheder") == temp_path
