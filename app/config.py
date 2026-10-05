from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str="Ai RAG API"
    VERSION: str="0.1.0"
    GEMINI_API_KEY:str
    PROJECT_VERSION:str="0.2.0"
    class Config:
        env_file=".env"
settings=Settings()