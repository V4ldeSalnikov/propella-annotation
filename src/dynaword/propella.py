import json
from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

SYSTEM_PROMPT = """Annotate the document. Any language; assess quality within its linguistic norms. Return only a JSON object. Do not include markdown, code fences, or explanatory text.
content_integrity: technical completeness (complete|mostly_complete|fragment|severely_degraded)
content_ratio: content vs navigation/boilerplate ratio (complete_content|mostly_content|mixed_content|mostly_navigation|minimal_content)
content_length: substantive words (substantial 2k+|moderate 500-2k|brief 100-500|minimal <100)
one_sentence_description: neutral ~10 word summary in English
content_type[]: functional purpose (analytical|instructional|reference|procedural|qa_structured|conversational|creative|transactional|boilerplate|news_report|opinion_editorial|review_critique|technical_documentation|specification_standard|legal_document|press_release|structured_data|source_code)
business_sector[]: industry domain (academic_research|education_sector|technology_software|hardware_electronics|healthcare_medical|pharmaceutical_biotech|financial_services|legal_services|government_public|manufacturing_industrial|mining_resources|chemicals_materials|energy_utilities|retail_commerce|wholesale_distribution|real_estate_construction|transportation_logistics|automotive_industry|telecommunications|media_entertainment|advertising_marketing|hospitality_tourism|agriculture_food|environmental_services|aerospace_defense|insurance_industry|nonprofit_ngo|consulting_professional|human_resources|security_cyber|gaming_industry|gambling_betting|travel_aviation|food_beverage_hospitality|consumer_goods|general_interest|other)
technical_content[]: specialized knowledge (code_heavy|math_heavy|scientific|data_heavy|engineering|basic_technical|non_technical)
content_quality: writing/presentation quality (excellent|good|adequate|poor|unacceptable)
information_density: signal vs padding (dense|adequate|moderate|thin|empty)
educational_value: teaching potential (high|moderate|basic|minimal|none)
reasoning_indicators: logical analysis depth (analytical|explanatory|basic_reasoning|minimal|none)
audience_level: assumed background (expert|advanced|general|beginner|youth|children)
commercial_bias: promotional influence (none|minimal|moderate|heavy|pure_marketing)
time_sensitivity: temporal decay (evergreen|slowly_changing|regularly_updating|time_sensitive)
content_safety: harmful content (safe|mild_concerns|nsfw|harmful|illegal)
pii_presence: private individual data (no_pii|contains_pii)
regional_relevance[]: geographic/cultural context (european|north_american|east_asian|south_asian|southeast_asian|middle_eastern|sub_saharan_african|latin_american|oceanian|central_asian|russian_sphere|global|culturally_neutral|indeterminate)
country_relevance[]: specific countries as ISO names, or supranational|none
"""

USER_PROMPT = """<start_of_document>
{content}
<end_of_document>
"""

TRUNCATION_TAG = "<truncated_content>"


class ContentIntegrity(str, Enum):
    complete = "complete"
    mostly_complete = "mostly_complete"
    fragment = "fragment"
    severely_degraded = "severely_degraded"


class ContentRatio(str, Enum):
    complete_content = "complete_content"
    mostly_content = "mostly_content"
    mixed_content = "mixed_content"
    mostly_navigation = "mostly_navigation"
    minimal_content = "minimal_content"


class ContentLength(str, Enum):
    substantial = "substantial"
    moderate = "moderate"
    brief = "brief"
    minimal = "minimal"


class ContentType(str, Enum):
    analytical = "analytical"
    instructional = "instructional"
    reference = "reference"
    procedural = "procedural"
    qa_structured = "qa_structured"
    conversational = "conversational"
    creative = "creative"
    transactional = "transactional"
    boilerplate = "boilerplate"
    news_report = "news_report"
    opinion_editorial = "opinion_editorial"
    review_critique = "review_critique"
    technical_documentation = "technical_documentation"
    specification_standard = "specification_standard"
    legal_document = "legal_document"
    press_release = "press_release"
    structured_data = "structured_data"
    source_code = "source_code"


class BusinessSector(str, Enum):
    academic_research = "academic_research"
    education_sector = "education_sector"
    technology_software = "technology_software"
    hardware_electronics = "hardware_electronics"
    healthcare_medical = "healthcare_medical"
    pharmaceutical_biotech = "pharmaceutical_biotech"
    financial_services = "financial_services"
    legal_services = "legal_services"
    government_public = "government_public"
    manufacturing_industrial = "manufacturing_industrial"
    mining_resources = "mining_resources"
    chemicals_materials = "chemicals_materials"
    energy_utilities = "energy_utilities"
    retail_commerce = "retail_commerce"
    wholesale_distribution = "wholesale_distribution"
    real_estate_construction = "real_estate_construction"
    transportation_logistics = "transportation_logistics"
    automotive_industry = "automotive_industry"
    telecommunications = "telecommunications"
    media_entertainment = "media_entertainment"
    advertising_marketing = "advertising_marketing"
    hospitality_tourism = "hospitality_tourism"
    agriculture_food = "agriculture_food"
    environmental_services = "environmental_services"
    aerospace_defense = "aerospace_defense"
    insurance_industry = "insurance_industry"
    nonprofit_ngo = "nonprofit_ngo"
    consulting_professional = "consulting_professional"
    human_resources = "human_resources"
    security_cyber = "security_cyber"
    gaming_industry = "gaming_industry"
    gambling_betting = "gambling_betting"
    travel_aviation = "travel_aviation"
    food_beverage_hospitality = "food_beverage_hospitality"
    consumer_goods = "consumer_goods"
    general_interest = "general_interest"
    other = "other"


class TechnicalContent(str, Enum):
    code_heavy = "code_heavy"
    math_heavy = "math_heavy"
    scientific = "scientific"
    data_heavy = "data_heavy"
    engineering = "engineering"
    basic_technical = "basic_technical"
    non_technical = "non_technical"


class ContentQuality(str, Enum):
    excellent = "excellent"
    good = "good"
    adequate = "adequate"
    poor = "poor"
    unacceptable = "unacceptable"


class InformationDensity(str, Enum):
    dense = "dense"
    adequate = "adequate"
    moderate = "moderate"
    thin = "thin"
    empty = "empty"


class EducationalValue(str, Enum):
    high = "high"
    moderate = "moderate"
    basic = "basic"
    minimal = "minimal"
    none = "none"


class ReasoningIndicators(str, Enum):
    analytical = "analytical"
    explanatory = "explanatory"
    basic_reasoning = "basic_reasoning"
    minimal = "minimal"
    none = "none"


class AudienceLevel(str, Enum):
    expert = "expert"
    advanced = "advanced"
    general = "general"
    beginner = "beginner"
    youth = "youth"
    children = "children"


class CommercialBias(str, Enum):
    none = "none"
    minimal = "minimal"
    moderate = "moderate"
    heavy = "heavy"
    pure_marketing = "pure_marketing"


class TimeSensitivity(str, Enum):
    evergreen = "evergreen"
    slowly_changing = "slowly_changing"
    regularly_updating = "regularly_updating"
    time_sensitive = "time_sensitive"


class ContentSafety(str, Enum):
    safe = "safe"
    mild_concerns = "mild_concerns"
    nsfw = "nsfw"
    harmful = "harmful"
    illegal = "illegal"


class PiiPresence(str, Enum):
    no_pii = "no_pii"
    contains_pii = "contains_pii"


class RegionalRelevance(str, Enum):
    european = "european"
    north_american = "north_american"
    east_asian = "east_asian"
    south_asian = "south_asian"
    southeast_asian = "southeast_asian"
    middle_eastern = "middle_eastern"
    sub_saharan_african = "sub_saharan_african"
    latin_american = "latin_american"
    oceanian = "oceanian"
    central_asian = "central_asian"
    russian_sphere = "russian_sphere"
    global_ = "global"
    culturally_neutral = "culturally_neutral"
    indeterminate = "indeterminate"


class AnnotationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_integrity: ContentIntegrity
    content_ratio: ContentRatio
    content_length: ContentLength
    one_sentence_description: str = Field(max_length=200)
    content_type: list[ContentType]
    business_sector: list[BusinessSector]
    technical_content: list[TechnicalContent]
    information_density: InformationDensity
    content_quality: ContentQuality
    audience_level: AudienceLevel
    commercial_bias: CommercialBias
    time_sensitivity: TimeSensitivity = Field(
        validation_alias=AliasChoices("time_sensitivity", "time_sensitive")
    )
    content_safety: ContentSafety
    educational_value: EducationalValue
    reasoning_indicators: ReasoningIndicators
    pii_presence: PiiPresence
    regional_relevance: list[RegionalRelevance]
    country_relevance: list[str]


def flatten_model_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema_copy = deepcopy(schema)
    definitions = schema_copy.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                extra = {key: value for key, value in node.items() if key != "$ref"}
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    definition_name = ref.rsplit("/", maxsplit=1)[-1]
                    resolved_definition = resolve(deepcopy(definitions[definition_name]))
                    resolved_extra = resolve(extra)
                    return {**resolved_definition, **resolved_extra}
            return {key: resolve(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema_copy)


def get_annotation_response_schema(
    flatten: bool = True,
    compact_whitespace: bool = True,
) -> dict[str, Any]:
    schema = AnnotationResponse.model_json_schema()

    if compact_whitespace:
        schema["x-guidance"] = {"whitespace_flexible": False}

    if flatten:
        schema = flatten_model_json_schema(schema)

    return json.loads(json.dumps(schema, separators=(",", ":"), ensure_ascii=False))


def truncate_content(content: str, max_content_chars: int = 50_000) -> str:
    if len(content) <= max_content_chars:
        return content
    return f"{content[:max_content_chars]}\n{TRUNCATION_TAG}"


def create_messages(
    document_text: str,
    max_content_chars: int = 50_000,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                content=truncate_content(document_text, max_content_chars)
            ),
        },
    ]
