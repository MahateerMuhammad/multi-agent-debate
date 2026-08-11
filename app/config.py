import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Agent Debate"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    VECTORSTORE_DIR: str = os.getenv("VECTORSTORE_DIR", "./data/vectorstore")
    
    class Config:
        case_sensitive = True

settings = Settings()
