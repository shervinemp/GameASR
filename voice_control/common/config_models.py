from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password: Optional[str] = None
    database: str = "neo4j"
    query_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)


class DatabaseConfig(BaseModel):
    neo4j: Neo4jConfig


class LLMModelsConfig(BaseModel):
    default: str
    extraction_heavy: str
    embedding: str


class LLMConfig(BaseModel):
    provider: str
    models: LLMModelsConfig
    providers: dict


class TTSConfig(BaseModel):
    provider: str
    weights_dir: str


class ASRConfig(BaseModel):
    provider: str
    weights_dir: str


class LLMServerConfig(BaseModel):
    auth_token: Optional[str] = Field(
        None,
        min_length=32,
        description="Authentication token for the RPC server.",
    )
    max_request_bytes: int = Field(default=65_536, ge=1_024, le=1_048_576)
    requests_per_minute: int = Field(default=60, ge=1, le=10_000)


class ToolClientConfig(BaseModel):
    auth_token: Optional[str] = Field(
        None,
        min_length=32,
        description="Authentication token for the tool client to connect to.",
    )


class ActiveLearningConfig(BaseModel):
    enabled: bool = False
    allow_web_context: bool = False
    review_required: bool = True
    review_queue_path: str = "data/pending_triplets.jsonl"
    max_triplets_per_answer: int = Field(default=20, ge=1, le=100)


class RAGRuntimeConfig(BaseModel):
    max_query_chars: int = Field(default=2_000, ge=64, le=16_384)
    top_k: int = Field(default=5, ge=1, le=20)
    reranker_input_limit: int = Field(default=20, ge=1, le=100)
    max_direct_context_tokens: int = Field(default=2_048, ge=256, le=65_536)
    retrieval_deadline_seconds: float = Field(default=8.0, ge=1.0, le=30.0)
    web_timeout_seconds: float = Field(default=4.0, ge=1.0, le=15.0)
    cache_ttl_seconds: float = Field(default=300.0, ge=0.0, le=86_400.0)
    cache_size: int = Field(default=128, ge=0, le=4_096)
    max_iterations: int = Field(default=3, ge=1, le=5)


class RAGConfig(BaseModel):
    runtime: RAGRuntimeConfig = Field(default_factory=RAGRuntimeConfig)
    active_learning: ActiveLearningConfig = Field(
        default_factory=ActiveLearningConfig
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    def model_post_init(self, __context):
        extra_keys = set(self.__pydantic_extra__ or {})
        if extra_keys:
            import logging
            logging.getLogger(__name__).warning(
                f"Unknown config keys ignored: {extra_keys}. "
                "Check for typos in your config.yaml."
            )

    database: DatabaseConfig
    llm: LLMConfig
    tts: TTSConfig
    asr: ASRConfig
    llm_server: Optional[LLMServerConfig] = None
    tool_client: Optional[ToolClientConfig] = None
    rpc_server: Optional[LLMServerConfig] = None
    tools_server: Optional[ToolClientConfig] = None
    rag: RAGConfig = Field(default_factory=RAGConfig)
