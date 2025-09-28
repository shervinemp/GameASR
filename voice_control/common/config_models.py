from pydantic import BaseModel, Field
from typing import Optional, Dict


class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password_env: str


class DatabaseConfig(BaseModel):
    neo4j: Neo4jConfig


class OllamaProviderConfig(BaseModel):
    model: str


class GeminiProviderConfig(BaseModel):
    api_key_env: str


class OpenAIProviderConfig(BaseModel):
    api_key_env: str


class LLMProvidersConfig(BaseModel):
    ollama: OllamaProviderConfig
    gemini: GeminiProviderConfig
    openai: OpenAIProviderConfig


class LLMModelsConfig(BaseModel):
    default: str
    extraction_heavy: str
    embedding: str


class LLMConfig(BaseModel):
    provider: str
    models: LLMModelsConfig
    providers: LLMProvidersConfig


class TTSConfig(BaseModel):
    provider: str
    model_dir: str


class ASRConfig(BaseModel):
    provider: str
    model_dir: str


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
    database: DatabaseConfig
    llm: LLMConfig
    tts: TTSConfig
    asr: ASRConfig
    llm_server: Optional[LLMServerConfig] = None
    tool_client: Optional[ToolClientConfig] = None
