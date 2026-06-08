from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # LLM Settings
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-5.5"          # Avaliador (LLM-as-Judge)
    TEMPERATURE: float = 0.0

    ANTHROPIC_API_KEY: str
    MODEL_CLAUDE: str =  "claude-sonnet-4-6"
    TEMPERATURE_CLAUDE: float = 0.4
    
    # Modelos Específicos por Agente
    MODEL_ORCHESTRATOR: str = "claude-sonnet-4-6"
    MODEL_COMPLIANCE: str = "gpt-5.4-mini"
    MODEL_AUDIO: str = "gpt-5.4-mini"

    # Pinecone Settings
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_CFM: str
    PINECONE_INDEX_POP: str
    
    # Security
    AGENTE_API_KEY: str
    ALLOWED_ORIGINS: str
    
    # Memory — PostgreSQL (Optional, falls back to in-memory if not set)
    DATABASE_URL: Optional[str] = None  # postgresql://user:password@host:5432/dbname

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
