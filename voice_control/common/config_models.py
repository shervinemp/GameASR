from pydantic import BaseModel, Field
from typing import Optional, Dict

class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password_env: str

class DatabaseConfig(BaseModel):
    neo4j: Neo4jConfig

class OllamaProviderConfig(BaseModel):
    base_url: str

class GeminiProviderConfig(BaseModel):
    api_key_env: str

class OpenAIProviderConfig(BaseModel):
    api_key_env: str

class LLMProvidersConfig(BaseModel):
    ollama: OllamaProviderConfig
    gemini: GeminiProviderConfig
    openai: OpenAIProviderConfig
    nemotron: Dict
    qwen: Dict

class LLMModelsConfig(BaseModel):
    default: str
    extraction_heavy: str
    embedding: str

class LLMConfig(BaseModel):
    default_provider: str
    models: LLMModelsConfig
    providers: LLMProvidersConfig

class KokoroTTSConfig(BaseModel):
    model_file: str
    voices_file: str

class TTSModelsConfig(BaseModel):
    kokoro: KokoroTTSConfig

class TTSConfig(BaseModel):
    provider: str
    model_dir: str
    models: TTSModelsConfig

class ASRConfig(BaseModel):
    provider: str
    model_dir: str

class RpcServerConfig(BaseModel):
    auth_token: Optional[str] = Field(None, description="Authentication token for the RPC server.")

class ToolsServerConfig(BaseModel):
    auth_token: Optional[str] = Field(None, description="Authentication token for the tools server.")

class AppConfig(BaseModel):
    database: DatabaseConfig
    llm: LLMConfig
    tts: TTSConfig
    asr: ASRConfig
    rpc_server: Optional[RpcServerConfig] = None
    tools_server: Optional[ToolsServerConfig] = None
