import os
from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac"}
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

SIGNATURES = {
    b"\xff\xfb": "mp3",
    b"\xff\xf3": "mp3",
    b"\xff\xf2": "mp3",
    b"\x49\x44\x33": "mp3",
    b"\x52\x49\x46\x46": "wav",
    b"\x66\x74\x79\x70": "m4a",
    b"\x4f\x67\x67\x53": "ogg",
    b"\x1a\x45\xdf\xa3": "webm",
    b"\x66\x4c\x61\x43": "flac",
    b"\x00\x00\x00\x20\x66\x74\x79\x70\x4d\x53\x4e\x56": "mp4",
    b"\x00\x00\x00\x18\x66\x74\x79\x70\x6d\x70\x34\x32": "m4a",
}


def _detect_format(data: bytes) -> str | None:
    for sig, fmt in SIGNATURES.items():
        if data[: len(sig)] == sig:
            return fmt
    return None


async def validate_audio_file(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "audio.mp3")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensão não permitida: {ext}. Permitidas: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    await file.seek(0)

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo muito grande ({len(content) / 1024 / 1024:.1f}MB). Máximo: {MAX_FILE_SIZE_MB}MB",
        )

    if content:
        detected = _detect_format(content[:16])
        if detected is None:
            raise HTTPException(
                status_code=400,
                detail="Formato de áudio não reconhecido. O arquivo pode estar corrompido.",
            )

    return ext


def validate_audio_file_path(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {file_path}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo muito grande ({size / 1024 / 1024:.1f}MB). Máximo: {MAX_FILE_SIZE_MB}MB",
        )

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensão não permitida: {ext}",
        )
