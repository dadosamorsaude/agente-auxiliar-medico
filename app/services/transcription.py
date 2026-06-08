from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from app.core.config import settings
from app.utils.audio import validate_audio_file_path
import logging
import os

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 25

def transcribe_audio(file_path: str) -> str:
    """
    Transcreve um arquivo de áudio usando o modelo Whisper da OpenAI.
    """
    validate_audio_file_path(file_path)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Arquivo muito grande ({file_size_mb:.1f}MB). "
            f"O Whisper suporta no máximo {MAX_FILE_SIZE_MB}MB."
        )

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )

        if not transcript or not transcript.strip():
            raise RuntimeError("Whisper retornou transcrição vazia.")

        return transcript.strip()

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except RateLimitError as e:
        logger.error(f"Rate limit excedido no Whisper: {e}")
        raise RuntimeError("Limite de requisições excedido. Tente novamente em alguns segundos.")
    except APITimeoutError:
        logger.error("Timeout na requisição ao Whisper")
        raise RuntimeError("A transcrição excedeu o tempo limite. Tente com um áudio menor.")
    except APIError as e:
        logger.error(f"Erro na API OpenAI (Whisper): {e}")
        raise RuntimeError(f"Erro no serviço de transcrição: {e.message}")
    except Exception as e:
        logger.error(f"Erro na transcrição Whisper: {e}")
        raise RuntimeError(f"Não foi possível transcrever o áudio: {str(e)}")
