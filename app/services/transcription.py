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

        # Pós-processamento de correção fonética e clínica via LLM
        try:
            logger.info("Iniciando pós-processamento de correção fonética com LLM...")
            correction_prompt = (
                "Você é um corretor ortográfico e clínico especializado em português do Brasil. "
                "Sua tarefa é ler uma transcrição de consulta médica (gerada por reconhecimento de voz) e "
                "corrigir erros fonéticos típicos de transcrição de áudio. "
                "Exemplos comuns de correção:\n"
                "- 'pressão vial' -> 'pressão arterial'\n"
                "- 'render os livros' -> 'reter líquidos'\n"
                "- 'suor para reidratar' -> 'soro para reidratar'\n"
                "- 'Velop' -> 'envelope'\n"
                "- 'Datetrona' -> 'Ondansetrona'\n"
                "- 'ajudar a aprender' -> 'ajudar a prender'\n"
                "- 'evite limites' -> 'evite alimentos'\n"
                "- 'uma testada de dispensa' -> 'um atestado de dispensa'\n"
                "- 'posto de atendimento' -> 'pronto atendimento'\n\n"
                "Regras estritas:\n"
                "1. Não adicione novos fatos clínicos, sintomas, exames, medicamentos ou prescrições que não estejam no áudio original.\n"
                "2. Mantenha o estilo e tom do discurso original do médico.\n"
                "3. Retorne APENAS o texto corrigido, sem qualquer introdução, explicação ou notas de rodapé."
            )

            correction_response = client.chat.completions.create(
                model=settings.MODEL_AUDIO,
                messages=[
                    {"role": "system", "content": correction_prompt},
                    {"role": "user", "content": f"Texto a corrigir:\n{transcript.strip()}"}
                ],
                temperature=0.0
            )
            corrected_text = correction_response.choices[0].message.content
            if corrected_text and corrected_text.strip():
                logger.info("Transcrição corrigida clinicamente com sucesso pelo LLM.")
                return corrected_text.strip()
        except Exception as e:
            logger.warning(f"Falha ao realizar pós-processamento de correção com LLM: {e}. Retornando transcrição original.")

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
