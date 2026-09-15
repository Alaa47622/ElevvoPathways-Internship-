from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    chroma_dir: str = Field(default="./storage/chroma", validation_alias="CHROMA_DIR")
    collection_name: str = Field(default="resume_entities", validation_alias="COLLECTION_NAME")
    top_k: int = Field(default=3, validation_alias="TOP_K")
    api_url: str = Field(default="http://localhost:8000", validation_alias="API_URL")

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
