import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from time import sleep

from dynaword.annotations import (
    PropellaAnnotator,
    annotate_dataset,
    create_token_bounded_messages,
    fits_context_window,
    get_partial_annotation_paths,
    get_temp_annotation_path,
    get_remote_annotation_paths,
    is_git_lfs_pointer,
    load_annotation_dataset,
    merge_annotation_parts,
    parse_annotation,
)
from dynaword.propella import TRUNCATION_TAG, create_messages


class StubAnnotator:
    def annotate(self, text: str) -> dict[str, str | list[str]]:
        return {
            "content_integrity": "complete",
            "content_ratio": "complete_content",
            "content_length": "minimal",
            "one_sentence_description": f"Description for {text}.",
            "content_type": ["conversational"],
            "business_sector": ["general_interest"],
            "technical_content": ["non_technical"],
            "information_density": "dense",
            "content_quality": "good",
            "audience_level": "general",
            "commercial_bias": "none",
            "time_sensitivity": "evergreen",
            "content_safety": "safe",
            "educational_value": "none",
            "reasoning_indicators": "none",
            "pii_presence": "no_pii",
            "regional_relevance": ["culturally_neutral"],
            "country_relevance": ["none"],
        }


class SlowStubAnnotator(StubAnnotator):
    def annotate(self, text: str) -> dict[str, str | list[str]]:
        if text == "slow":
            sleep(0.02)
        if text == "medium":
            sleep(0.01)
        return super().annotate(text)


class StubTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> list[int]:
        assert add_generation_prompt is True
        assert tokenize is True
        size = sum(len(message["content"]) for message in conversation)
        return [0] * size


class StubResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class StubCompletions:
    def __init__(self, contents: list[str | None]) -> None:
        self.contents = list(contents)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return StubResponse(self.contents.pop(0))


class StubClient:
    def __init__(self, contents: list[str | None]) -> None:
        self.chat = type("Chat", (), {"completions": StubCompletions(contents)})()


class StubRepoApi:
    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        assert repo_id == "danish-foundation-models/dynaword-annotations"
        assert repo_type == "dataset"
        return [
            "annotations/gutenberg/ellamind--propella-1-4b.parquet",
            "annotations/nota/ellamind--propella-1-4b.parquet",
        ]


def test_parse_annotation():
    response = """{"content_integrity":"complete","content_ratio":"complete_content","content_length":"minimal","one_sentence_description":"A short question and answer.","content_type":["qa_structured"],"business_sector":["general_interest"],"technical_content":["non_technical"],"information_density":"dense","content_quality":"good","audience_level":"general","commercial_bias":"none","time_sensitivity":"evergreen","content_safety":"safe","educational_value":"none","reasoning_indicators":"none","pii_presence":"no_pii","regional_relevance":["culturally_neutral"],"country_relevance":["none"]}"""
    parsed = parse_annotation(response)
    assert parsed["time_sensitivity"] == "evergreen"


def test_get_remote_annotation_paths():
    assert get_remote_annotation_paths(StubRepoApi()) == {
        "annotations/gutenberg/ellamind--propella-1-4b.parquet",
        "annotations/nota/ellamind--propella-1-4b.parquet",
    }


def test_annotator_retries_after_truncated_json():
    valid_response = """{"content_integrity":"complete","content_ratio":"complete_content","content_length":"minimal","one_sentence_description":"A short question and answer.","content_type":["qa_structured"],"business_sector":["general_interest"],"technical_content":["non_technical"],"information_density":"dense","content_quality":"good","audience_level":"general","commercial_bias":"none","time_sensitivity":"evergreen","content_safety":"safe","educational_value":"none","reasoning_indicators":"none","pii_presence":"no_pii","regional_relevance":["culturally_neutral"],"country_relevance":["none"]}"""
    client = StubClient(
        [
            '{"content_integrity":"complete"',
            valid_response,
        ]
    )
    annotator = PropellaAnnotator(
        api_base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model_name="ellamind/propella-1-4b",
        client=client,
        tokenizer=StubTokenizer(),
    )

    annotation = annotator.annotate("hej")

    assert annotation["time_sensitivity"] == "evergreen"
    calls = client.chat.completions.calls
    assert [call["max_tokens"] for call in calls] == [512, 1024]
    assert calls[0]["messages"] == calls[1]["messages"]


def test_fits_context_window():
    assert fits_context_window(prompt_tokens=7000, max_output_tokens=512) is True
    assert fits_context_window(prompt_tokens=7500, max_output_tokens=512) is False


def test_create_messages_truncates_content():
    messages = create_messages("abcdef", max_content_chars=3)
    assert messages[1]["content"].startswith("<start_of_document>\nabc")
    assert TRUNCATION_TAG in messages[1]["content"]


def test_create_token_bounded_messages_truncates_to_fit_budget():
    empty_prompt_tokens = sum(
        len(message["content"])
        for message in create_messages("", max_content_chars=0)
    )
    messages = create_token_bounded_messages(
        "a" * 500,
        tokenizer=StubTokenizer(),
        max_input_tokens=empty_prompt_tokens + 100,
    )

    assert sum(len(message["content"]) for message in messages) <= (
        empty_prompt_tokens + 100
    )
    assert TRUNCATION_TAG in messages[1]["content"]


def test_annotate_dataset_writes_parquet(tmp_path):
    dataset_path = tmp_path / "demo.parquet"
    output_path = tmp_path / "annotations.parquet"

    ds = Dataset.from_dict({"id": ["row-1", "row-2"], "text": ["hej", "verden"]})
    ds.to_parquet(str(dataset_path))

    annotate_dataset(
        dataset_name="demo",
        dataset_path=dataset_path,
        output_path=output_path,
        annotator=StubAnnotator(),
    )

    rows = pq.read_table(output_path).to_pylist()
    assert [row["id"] for row in rows] == ["row-1", "row-2"]
    assert rows[0]["one_sentence_description"] == "Description for hej."
    assert rows[1]["content_type"] == ["conversational"]


def test_annotate_dataset_preserves_order_with_concurrent_annotation(tmp_path):
    dataset_path = tmp_path / "demo.parquet"
    output_path = tmp_path / "annotations.parquet"

    ds = Dataset.from_dict(
        {
            "id": ["row-1", "row-2", "row-3"],
            "text": ["slow", "fast", "medium"],
        }
    )
    ds.to_parquet(str(dataset_path))

    annotate_dataset(
        dataset_name="demo",
        dataset_path=dataset_path,
        output_path=output_path,
        annotator=SlowStubAnnotator(),
    )

    rows = pq.read_table(output_path).to_pylist()
    assert [row["id"] for row in rows] == ["row-1", "row-2", "row-3"]
    assert rows[2]["one_sentence_description"] == "Description for medium."


def test_annotate_dataset_resumes_from_partial_parquet(tmp_path):
    dataset_path = tmp_path / "demo.parquet"
    output_path = tmp_path / "annotations.parquet"

    ds = Dataset.from_dict(
        {
            "id": ["row-1", "row-2", "row-3"],
            "text": ["hej", "verden", "igen"],
        }
    )
    ds.to_parquet(str(dataset_path))

    temp_path = get_temp_annotation_path(output_path)
    first_record = {
        "id": "row-1",
        **StubAnnotator().annotate("hej"),
    }
    pq.write_table(pa.Table.from_pylist([first_record]), temp_path)

    annotate_dataset(
        dataset_name="demo",
        dataset_path=dataset_path,
        output_path=output_path,
        annotator=StubAnnotator(),
    )

    rows = pq.read_table(output_path).to_pylist()
    assert [row["id"] for row in rows] == ["row-1", "row-2", "row-3"]
    assert not temp_path.exists()


def test_merge_annotation_parts(tmp_path):
    output_path = tmp_path / "annotations.parquet"
    temp_path = get_temp_annotation_path(output_path)
    part_path = output_path.with_suffix(".part1.parquet")

    pq.write_table(pa.Table.from_pylist([{"id": "row-1", "one_sentence_description": "a"}]), temp_path)
    pq.write_table(pa.Table.from_pylist([{"id": "row-2", "one_sentence_description": "b"}]), part_path)

    merge_annotation_parts([temp_path, part_path], output_path)

    rows = pq.read_table(output_path).to_pylist()
    assert [row["id"] for row in rows] == ["row-1", "row-2"]
    assert get_partial_annotation_paths(output_path) == []


def test_is_git_lfs_pointer(tmp_path):
    pointer_path = tmp_path / "pointer.parquet"
    pointer_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 123\n"
    )

    assert is_git_lfs_pointer(pointer_path) is True


def test_load_annotation_dataset_falls_back_to_hub_for_lfs_pointer(
    tmp_path, monkeypatch
):
    dataset_path = tmp_path / "gutenberg.parquet"
    dataset_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 123\n"
    )

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset.from_dict({"id": ["row-1"], "text": ["hej"]})

    monkeypatch.setattr("dynaword.annotations.load_dataset", fake_load_dataset)

    dataset = load_annotation_dataset("gutenberg", dataset_path)

    assert len(dataset) == 1
    assert calls == [
        (
            ("danish-foundation-models/danish-dynaword", "gutenberg"),
            {"split": "train"},
        )
    ]
