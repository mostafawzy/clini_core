from pydantic_settings import BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    groq_api_key: SecretStr | None = None

    # RAG
    groq_model: str = ""

       
    

    model_config = {
        "env_file": ".env"
    }


def get_settings():
    settings = Settings()

    if settings.groq_api_key is None:
        raise ValueError("GROQ_API_KEY is not set")

    return settings


