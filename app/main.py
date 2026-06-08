from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.metrics import router as metrics_router
from app.api.audio import router as audio_router
from app.api.voice import router as voice_router
from app.api.ws import router as ws_router
from app.core.logger import logger
from app.utils.rate_limiter import rate_limiter, AUDIO_RATE_LIMIT, AUDIO_WINDOW, CHAT_VOICE_RATE_LIMIT, CHAT_VOICE_WINDOW
from contextlib import asynccontextmanager
import time
import os
import asyncio
import shutil
from dotenv import load_dotenv

# Carrega arquivos .env pro os.environ (essencial pro LangSmith enxergar as chaves no ambiente)
load_dotenv(override=True)

TEMP_DIR = "temp_audios"
MAX_AGE_HOURS = 1


async def _cleanup_orphan_audios():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        return
    now = time.time()
    removed = 0
    for fname in os.listdir(TEMP_DIR):
        fpath = os.path.join(TEMP_DIR, fname)
        if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) / 3600 > MAX_AGE_HOURS:
            try:
                os.remove(fpath)
                removed += 1
            except Exception:
                pass
    if removed:
        logger.info(f"Cleanup inicial: removidos {removed} arquivos órfãos")


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(1800)
        await _cleanup_orphan_audios()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _cleanup_orphan_audios()
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Auxiliar Medico AI Agent",
    version="0.1.0",
    description="Agente de transcrição e auditoria de consultas clínicas.",
    lifespan=lifespan,
)

from app.core.config import settings

# Parse de domínios permitidos via variável de ambiente (separados por vírgula)
allowed_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]

# Garante a inclusão dos domínios padrão do Lovable (produção, preview, sandbox, local)
lovable_origins = [
    "https://aux-care-chat.lovable.app",
    "https://id-preview--aa62bec8-ff59-4d10-8fa9-d8f116ab1869.lovable.app",
    "https://aa62bec8-ff59-4d10-8fa9-d8f116ab1869.lovableproject.com",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000"
]

for origin in lovable_origins:
    if origin not in allowed_origins:
        allowed_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

AUDIO_ROUTES = {"/audio/upload", "/chat/voice", "/ws/voice"}

VOICE_RATE_LIMIT = 10
VOICE_WINDOW = 60


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Completed {request.method} {request.url.path} with status {response.status_code} in {process_time:.3f}s")

    return response


@app.middleware("http")
async def rate_limit_audio(request: Request, call_next):
    if request.url.path in AUDIO_ROUTES:
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("x-api-key", "")
        rate_key = f"audio:{client_ip}:{api_key}"

        await rate_limiter.check(rate_key, VOICE_RATE_LIMIT, VOICE_WINDOW)

    return await call_next(request)





app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(audio_router)
app.include_router(voice_router)
app.include_router(ws_router)


@app.get("/")
def home():
    """Health check endpoint for Render monitoring."""
    logger.info("Health check endpoint called.")
    return {
        "status": "ok",
        "agent": "auxiliar-medico",
        "version": "0.1.0",
        "environment": "production" if os.getenv("RENDER") else "development"
    }