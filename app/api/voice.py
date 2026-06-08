from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Security
from fastapi.responses import StreamingResponse
from app.agent.orchestrator import run_agent
from app.api.security import get_api_key
from app.services.transcription import transcribe_audio
from app.utils.audio import validate_audio_file
from app.core.logger import logger
import os
import uuid
import json
from typing import AsyncGenerator

router = APIRouter(prefix="/chat", tags=["voice"])

UPLOAD_DIR = "temp_audios"


@router.post("/voice")
async def chat_voice(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Security(get_api_key),
):
    """
    Endpoint unificado para Lovable:
    1. Recebe áudio.
    2. Transcreve usando Whisper.
    # 3. Processa o texto com o Agente Auxiliar Médico
    # 4. Retorna a resposta final.
    """
    logger.info(f"Recebido pedido de chat por voz | user_id: {user_id}")

    ext = await validate_audio_file(file)
    temp_filename = f"voice_{uuid.uuid4()}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        async def process_voice() -> AsyncGenerator[str, None]:
            # 1. Transcrição
            yield f"data: {json.dumps({'type': 'status', 'text': 'Transcrevendo áudio...'}, ensure_ascii=False)}\n\n"
            logger.info(f"Transcrevendo áudio temporário: {temp_filename}")

            try:
                transcribed_text = transcribe_audio(temp_path)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Erro ao transcrever áudio: {str(e)}'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                os.remove(temp_path)
                return

            if not transcribed_text:
                yield f"data: {json.dumps({'type': 'error', 'text': 'Não foi possível transcrever o áudio.'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                os.remove(temp_path)
                return

            logger.info(f"Transcrição concluída: {transcribed_text[:50]}...")
            yield f"data: {json.dumps({'type': 'transcription', 'text': transcribed_text}, ensure_ascii=False)}\n\n"

            # 2. Análise
            yield f"data: {json.dumps({'type': 'status', 'text': 'Realizando análise de conformidade clínica...'}, ensure_ascii=False)}\n\n"

            full_query = (
                "Com base na transcrição abaixo, realize as seguintes tarefas:\n"
                "1. Estruture o texto nos campos: ANAMNESE, CONDUTA, HIPÓTESE DIAGNÓSTICA e CID-10.\n"
                "2. Realize uma auditoria de conformidade clínica baseada nas normas do CFM e POPs internos, "
                "verificando se os campos estruturados atendem aos critérios de qualidade do Auxiliar Médico.\n\n"
                f"Transcrição:\n{transcribed_text}"
            )

            full_response = ""
            async for chunk in run_agent(user_id, full_query, stream=False):
                if chunk:
                    full_response += chunk

            yield f"data: {json.dumps({'type': 'response', 'transcription': transcribed_text, 'text': full_response}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            os.remove(temp_path)

        return StreamingResponse(
            process_voice(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        logger.error(f"Erro no processamento de voz: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Erro interno no processamento de voz.")
