"""
Persistência das avaliações de acurácia no PostgreSQL.

Cria e mantém a tabela `evaluation_logs` no mesmo banco já usado
para o histórico de chat (Supabase/PostgreSQL).
Com fallback para lista in-memory se DATABASE_URL não estiver configurado.
"""

import json
import logging
from datetime import datetime

import psycopg
from psycopg import AsyncConnection

from app.core.config import settings

logger = logging.getLogger(__name__)

# Fallback in-memory quando não há banco configurado
_memory_store: list[dict] = []

_table_created = False

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_logs (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    question        TEXT NOT NULL,
    response        TEXT NOT NULL,
    raw_data        JSONB,
    score           INTEGER,
    approved        BOOLEAN,
    errors          JSONB,
    justification   TEXT,
    breakdown       JSONB,
    model           TEXT
);
"""

INSERT_SQL = """
INSERT INTO evaluation_logs
    (user_id, created_at, question, response, raw_data, score, approved, errors, justification, breakdown, model)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""


async def _ensure_table(conn: AsyncConnection) -> None:
    """Cria a tabela se não existir (executado uma vez por processo)."""
    global _table_created
    if _table_created:
        return
    try:
        async with conn.cursor() as cur:
            await cur.execute(CREATE_TABLE_SQL)
        await conn.commit()
        _table_created = True
        logger.info("Tabela 'evaluation_logs' verificada/criada com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao criar tabela evaluation_logs: {e}")


async def save_evaluation(
    user_id: str,
    question: str,
    response: str,
    evaluation: dict,
) -> None:
    """
    Persiste o resultado de uma avaliação.
    Usa PostgreSQL se disponível, caso contrário armazena em memória.
    """
    record = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "question": question,
        "response": response,
        "raw_data": [], # Mantemos vazio no Auxiliar Médico (sem Athena)
        "score": evaluation.get("score", 0),
        "approved": evaluation.get("aprovado", False),
        "errors": evaluation.get("erros_encontrados", []),
        "justification": evaluation.get("justificativa", ""),
        "breakdown": {
            "fidelidade_clinica": evaluation.get("fidelidade_clinica", 0),
            "estruturacao_completude": evaluation.get("estruturacao_completude", 0),
            "aplicacao_normativa": evaluation.get("aplicacao_normativa", 0),
        },
        "model": evaluation.get("model", settings.MODEL_NAME),
    }

    if settings.DATABASE_URL:
        try:
            async with await AsyncConnection.connect(settings.DATABASE_URL) as conn:
                await _ensure_table(conn)

                async with conn.cursor() as cur:
                    await cur.execute(
                        INSERT_SQL,
                        (
                            record["user_id"],
                            record["created_at"],
                            record["question"],
                            record["response"],
                            json.dumps(record["raw_data"]),
                            record["score"],
                            record["approved"],
                            json.dumps(record["errors"]),
                            record["justification"],
                            json.dumps(record["breakdown"]),
                            record["model"],
                        ),
                    )
                await conn.commit()

            logger.info(f"Avaliação salva no PostgreSQL | score={record['score']}")
            return
        except Exception as e:
            logger.warning(f"Falha ao salvar avaliação no PostgreSQL, usando in-memory: {e}")

    # Fallback: in-memory (sem persistência entre reinicializações)
    _memory_store.append(record)
    logger.info(f"Avaliação salva in-memory | score={record['score']} | total={len(_memory_store)}")


async def get_evaluation_summary() -> dict:
    """Retorna o resumo agregado de todas as avaliações."""
    if settings.DATABASE_URL:
        try:
            async with await AsyncConnection.connect(settings.DATABASE_URL) as conn:
                await _ensure_table(conn)

                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT
                            COUNT(*)                            AS total,
                            ROUND(AVG(score)::numeric, 1)       AS avg_score,
                            ROUND(
                                100.0 * SUM(CASE WHEN approved THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                                1
                            )                                   AS approved_rate,
                            ROUND(
                                AVG(score) FILTER (
                                    WHERE created_at >= NOW() - INTERVAL '7 days'
                                )::numeric, 1
                            )                                   AS avg_score_7d
                        FROM evaluation_logs;
                    """)
                    row = await cur.fetchone()

                    await cur.execute("""
                        SELECT elem
                        FROM evaluation_logs,
                             jsonb_array_elements_text(errors) AS elem
                        WHERE approved = false
                        ORDER BY created_at DESC
                        LIMIT 50;
                    """)
                    error_rows = await cur.fetchall()

                common_errors = _top_errors([r[0] for r in error_rows])

                return {
                    "total_evaluations": row[0] or 0,
                    "avg_score": float(row[1]) if row[1] else 0.0,
                    "approved_rate": float(row[2]) if row[2] else 0.0,
                    "avg_score_last_7d": float(row[3]) if row[3] else 0.0,
                    "common_errors": common_errors,
                }
        except Exception as e:
            logger.error(f"Erro ao buscar resumo do PostgreSQL: {e}")

    # Fallback: calcular de _memory_store
    return _summary_from_memory()


async def get_evaluation_history(limit: int = 20) -> list[dict]:
    """Retorna as últimas avaliações."""
    if settings.DATABASE_URL:
        try:
            async with await AsyncConnection.connect(settings.DATABASE_URL) as conn:
                await _ensure_table(conn)

                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT id, user_id, created_at, question, score, approved,
                               errors, justification, breakdown
                        FROM evaluation_logs
                        ORDER BY created_at DESC
                        LIMIT %s;
                    """, (limit,))
                    rows = await cur.fetchall()
                    cols = ["id", "user_id", "created_at", "question", "score",
                            "approved", "errors", "justification", "breakdown"]

                return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error(f"Erro ao buscar histórico do PostgreSQL: {e}")

    # Fallback in-memory
    return sorted(_memory_store, key=lambda x: x["created_at"], reverse=True)[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ──────────────────────────────────────────────────────────────────────────────

def _top_errors(errors: list[str], n: int = 5) -> list[str]:
    """Retorna os N erros mais frequentes."""
    from collections import Counter
    return [err for err, _ in Counter(errors).most_common(n)]


def _summary_from_memory() -> dict:
    data = _memory_store
    if not data:
        return {"total_evaluations": 0, "avg_score": 0.0,
                "approved_rate": 0.0, "avg_score_last_7d": 0.0, "common_errors": []}

    scores = [d["score"] for d in data]
    approved = [d["approved"] for d in data]
    all_errors = [e for d in data for e in d.get("errors", [])]

    return {
        "total_evaluations": len(data),
        "avg_score": round(sum(scores) / len(scores), 1),
        "approved_rate": round(100 * sum(approved) / len(approved), 1),
        "avg_score_last_7d": round(sum(scores) / len(scores), 1),
        "common_errors": _top_errors(all_errors),
    }
