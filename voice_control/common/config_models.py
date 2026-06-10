from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict


class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password: Optional[str] = None


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
        None, description="Authentication token for the LLM server to host."
    )


class ToolClientConfig(BaseModel):
    auth_token: Optional[str] = Field(
        None,
        description="Authentication token for the tool client to connect to.",
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
