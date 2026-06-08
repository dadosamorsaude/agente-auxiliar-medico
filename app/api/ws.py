from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.agent.orchestrator import run_agent
from app.services.transcription import transcribe_audio
from app.core.config import settings
from app.core.logger import logger
import os
import uuid
import json

router = APIRouter(tags=["voice-streaming"])

UPLOAD_DIR = "temp_audios"

AUTH_TIMEOUT = 10


@router.websocket("/ws/voice")
async def websocket_voice_endpoint(
    websocket: WebSocket,
):
    """
    WebSocket para streaming de voz com autenticação via mensagem inicial.
    Protocolo:
    1. Auth: { "type": "auth", "api_key": "..." }
    2. Handshake: { "type": "start", "mime_type": "...", "sample_rate": ... }
    3. Data: Binary chunks (WebM/Opus)
    4. Finish: { "type": "stop" }
    """
    await websocket.accept()

    # 1. Autenticação via primeira mensagem
    authenticated = False
    try:
        msg = await websocket.receive_text()
        data = json.loads(msg)
        if data.get("type") == "auth" and data.get("api_key") == settings.AGENTE_API_KEY:
            authenticated = True
            await websocket.send_json({"type": "auth_ok"})
        else:
            await websocket.close(code=4003)
            return
    except (json.JSONDecodeError, Exception):
        await websocket.close(code=4003)
        return

    session_id = str(uuid.uuid4())
    logger.info(f"Conexão Voice-WS autenticada | session_id: {session_id}")

    temp_filename = f"stream_{session_id}.webm"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    audio_file = None
    is_recording = False

    try:
        while True:
            message = await websocket.receive()

            # A) TRATAMENTO DE BINÁRIO (CHUNKS DE ÁUDIO)
            if "bytes" in message:
                if is_recording and audio_file:
                    audio_file.write(message["bytes"])
                continue

            # B) TRATAMENTO DE TEXTO (COMANDOS JSON)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "start":
                        logger.info(f"Iniciando gravação | session_id: {session_id}")
                        audio_file = open(temp_path, "wb")
                        is_recording = True
                        await websocket.send_json({"type": "partial", "text": "Gravando áudio..."})

                    elif msg_type == "stop":
                        if not is_recording or not audio_file:
                            continue

                        logger.info(f"Finalizando gravação | session_id: {session_id}")
                        is_recording = False
                        audio_file.close()
                        audio_file = None

                        await websocket.send_json({"type": "partial", "text": "Processando transcrição..."})

                        # 1. Transcrição com Whisper
                        transcribed_text = transcribe_audio(temp_path)

                        if not transcribed_text:
                            await websocket.send_json({"type": "error", "message": "Não foi possível transcrever o áudio."})
                            continue

                        await websocket.send_json({"type": "final", "text": transcribed_text})

                        # 2. Análise Clínica Automática (Auxiliar Médico)
                        await websocket.send_json({"type": "partial", "text": "Realizando análise de conformidade clínica..."})

                        full_query = (
                            "Com base na transcrição abaixo, realize as seguintes tarefas:\n"
                            "1. Estruture o texto nos campos: ANAMNESE, CONDUTA, HIPÓTESE DIAGNÓSTICA e CID-10.\n"
                            "2. Realize uma auditoria de conformidade clínica baseada nas normas do CFM e POPs internos, "
                            "verificando se os campos estruturados atendem aos critérios de qualidade do Auxiliar Médico.\n\n"
                            f"Transcrição:\n{transcribed_text}"
                        )

                        full_response = ""
                        async for chunk in run_agent(session_id, full_query, stream=True):
                            if chunk:
                                full_response += chunk
                                await websocket.send_json({"type": "partial", "text": chunk})

                        await websocket.send_json({
                            "type": "final",
                            "text": full_response,
                            "is_analysis": True
                        })

                except json.JSONDecodeError:
                    logger.warning(f"Recebido texto não-JSON no WS: {message['text']}")
                    continue

    except WebSocketDisconnect:
        logger.info(f"Voice-WS desconectado | session_id: {session_id}")
    except Exception as e:
        logger.error(f"Erro no Voice-WS: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "Erro interno no processamento de voz."})
        except:
            pass
    finally:
        if audio_file:
            audio_file.close()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
