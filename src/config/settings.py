import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Project Identity
    PROJECT_NAME: str = "AI Research Scientist Assistant"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Directories
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_METADATA_DIR: Path = PROJECT_ROOT / "data" / "raw" / "metadata"
    RAW_PDF_DIR: Path = PROJECT_ROOT / "data" / "raw" / "pdf"
    PROCESSED_TEXT_DIR: Path = PROJECT_ROOT / "data" / "processed" / "text"
    PROCESSED_JSON_DIR: Path = PROJECT_ROOT / "data" / "processed" / "json"
    PROCESSED_CHUNKS_DIR: Path = PROJECT_ROOT / "data" / "processed" / "chunks"
    EVALUATION_DIR: Path = PROJECT_ROOT / "data" / "evaluation"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    QDRANT_STORAGE_DIR: Path = PROJECT_ROOT / "storage" / "qdrant"
    DATABASE_PATH: Path = PROJECT_ROOT / "storage" / "metadata.db"

    # API Keys & Discovery Credentials
    OPENALEX_API_KEY: str = Field(default="")
    OPENALEX_EMAIL: str = Field(default="researcher@example.com")
    SEMANTIC_SCHOLAR_API_KEY: str = Field(default="")

    # Vector Store (Qdrant)
    QDRANT_URL: str = Field(default="")  # If empty, embedded local disk mode is used
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_COLLECTION: str = Field(default="research_chunks")
    VECTOR_DIMENSION: int = Field(default=384)  # Default for bge-small or all-MiniLM; 1024 for bge-m3

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5")
    EMBEDDING_DEVICE: str = Field(default="cpu")
    EMBEDDING_BATCH_SIZE: int = Field(default=32)

    # Chunking Configuration
    CHUNK_SIZE_TOKENS: int = Field(default=600)
    CHUNK_OVERLAP_TOKENS: int = Field(default=100)

    # Retrieval Configuration
    DEFAULT_TOP_K: int = Field(default=5)
    HYBRID_ALPHA: float = Field(default=0.7)  # 0.7 semantic, 0.3 keyword BM25

    # LLM Generation (Ollama)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3:latest")
    LLM_TIMEOUT: float = Field(default=180.0)

    # API Server
    API_HOST: str = Field(default="127.0.0.1")
    API_PORT: int = Field(default=8000)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.RAW_METADATA_DIR,
            self.RAW_PDF_DIR,
            self.PROCESSED_TEXT_DIR,
            self.PROCESSED_JSON_DIR,
            self.PROCESSED_CHUNKS_DIR,
            self.EVALUATION_DIR,
            self.STORAGE_DIR,
            self.QDRANT_STORAGE_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
