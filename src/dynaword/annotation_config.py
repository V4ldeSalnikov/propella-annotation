from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotationConfig:
    dataset_repo_id: str = "danish-foundation-models/danish-dynaword"
    model_name: str = "ellamind/propella-1-4b"
    api_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    annotation_concurrency: int = 8
    annotation_batch_size: int = 32
    max_context_tokens: int = 8_192
    max_content_chars: int = 50_000
    max_new_tokens: int = 512
    retry_max_new_tokens: int = 1_024
    prompt_token_margin: int = 256
    upload_repo_id: str = "danish-foundation-models/dynaword-annotations"


DEFAULT_ANNOTATION_CONFIG = AnnotationConfig()
