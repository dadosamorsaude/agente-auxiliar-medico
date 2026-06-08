"""
Agente Avaliador de Acurácia — LLM-as-Judge

Compara a resposta do Auxiliar Médico com a transcrição original e as normas do RAG,
gerando um score de 0-100 com justificativa detalhada dos erros encontrados.
"""

import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langsmith import traceable

from app.core.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt do Avaliador
# ──────────────────────────────────────────────────────────────────────────────

EVALUATOR_SYSTEM_PROMPT = """
Você é um avaliador especializado em auditoria de prontuários médicos e transcrições clínicas.

Sua única função é avaliar se a RESPOSTA DO AGENTE reflete com precisão, clareza e fidelidade a TRANSCRIÇÃO DA CONSULTA (texto/áudio original) fornecida E aplica corretamente as DIRETRIZES NORMATIVAS CFM/POPs recuperadas.

## Critérios de Avaliação

1. **Fidelidade Clínica (0–40 pts)**
   - O agente alterou, omitiu de forma prejudicial ou inventou (alucinou) sintomas, medicamentos, diagnósticos ou dados vitais não mencionados no texto original da consulta?
   - O agente foi excessivamente assertivo ao sugerir hipóteses em vez de colocá-las como sugestões do sistema a validar pelo médico?

2. **Qualidade da Estruturação e Completude (0–30 pts)**
   - A resposta organizou as seções ANAMNESE, CONDUTA, HIPÓTESE DIAGNÓSTICA, CID-10 e ORIENTAÇÃO de forma clara e abrangendo os fatos principais?

3. **Aplicação Normativa e Sugestões (0–30 pts)**
   - O agente usou corretamente as diretrizes CFM/POPs do RAG para apontar pontos omissos e falhas de conformidade no prontuário?
   - As recomendações de melhorias fornecidas pelo agente ao médico são pertinentes, seguras e úteis?
   - Se não houver diretriz normativa consultada, atribua 30 automaticamente.

## Regras
- Avalie somente com base na transcrição original da consulta e diretrizes fornecidas, não em conhecimento prévio.
- Se a transcrição estiver vazia, retorne score 0 com justificativa.
- Seja objetivo e específico nos erros encontrados.
- Responda APENAS com o JSON solicitado, sem texto adicional.
"""

EVALUATOR_USER_TEMPLATE = """
## Transcrição Original da Consulta (ou Pergunta):
{raw_transcription}

## Contexto Normativo Recuperado (CFM / POPs):
{rag_context}

## Pergunta/Comando do Usuário:
{user_question}

## Resposta do Agente (Auxiliar Médico):
{agent_response}

## Histórico da Conversa (Memória):
{chat_history}

## Avalie e responda APENAS em JSON:
{{
  "score": <inteiro 0-100>,
  "fidelidade_clinica": <inteiro 0-40>,
  "estruturacao_completude": <inteiro 0-30>,
  "aplicacao_normativa": <inteiro 0-30>,
  "aprovado": <true se score >= 70, false caso contrário>,
  "erros_encontrados": [<lista de strings descrevendo cada erro específico>],
  "justificativa": "<resumo objetivo da avaliação em 2-3 frases>"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Avaliador Principal
# ──────────────────────────────────────────────────────────────────────────────

@traceable(name="evaluator_llm")
def _get_evaluator_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY,
        metadata={"run": "evaluator"},
    )


@traceable(name="evaluate_response", run_type="chain")
async def evaluate_response(
    user_question: str,
    agent_response: str,
    rag_context: list[dict] | None = None,
    chat_history: str = "",
) -> dict:
    """Avalia a acurácia da resposta do Auxiliar Médico via LLM-as-Judge."""
    if not user_question and not rag_context:
        logger.warning("Avaliador: nenhuma pergunta de entrada nem RAG disponível.")
        return _empty_evaluation("Nenhuma pergunta ou RAG disponível para avaliar.")

    # Formata contexto normativo do RAG
    if rag_context:
        rag_parts = []
        for item in rag_context:
            source = item.get("source", "Desconhecido")
            query = item.get("query", "")
            chunks = "\n---\n".join(item.get("chunks", []))
            rag_parts.append(f"[{source}] Query: '{query}'\n{chunks}")
        rag_context_str = "\n\n".join(rag_parts)
    else:
        rag_context_str = "Nenhuma diretriz normativa foi consultada nesta resposta."

    user_message = EVALUATOR_USER_TEMPLATE.format(
        raw_transcription=user_question,
        rag_context=rag_context_str,
        user_question=user_question,
        agent_response=agent_response,
        chat_history=chat_history or "Nenhum histórico anterior.",
    )

    try:
        llm = _get_evaluator_llm()
        response = await llm.ainvoke([
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])

        raw_content = response.content.strip()

        # Remove possíveis blocos de código markdown (```json ... ```)
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]

        evaluation = json.loads(raw_content)
        evaluation["evaluated_at"] = datetime.utcnow().isoformat()
        evaluation["model"] = settings.MODEL_NAME
        evaluation["had_rag_context"] = bool(rag_context)
        evaluation["had_athena_data"] = False

        logger.info(
            f"Avaliação concluída | score={evaluation.get('score')} "
            f"| aprovado={evaluation.get('aprovado')}"
        )
        return evaluation

    except json.JSONDecodeError as e:
        logger.error(f"Avaliador: resposta do LLM não é JSON válido: {e}")
        return _empty_evaluation(f"Falha ao parsear resposta do avaliador: {e}")

    except Exception as e:
        logger.exception("Avaliador: erro ao invocar LLM avaliador")
        return _empty_evaluation(f"Erro interno no avaliador: {e}")


def _empty_evaluation(reason: str) -> dict:
    """Retorna uma avaliação vazia com score 0 quando não é possível avaliar."""
    return {
        "score": 0,
        "fidelidade_clinica": 0,
        "estruturacao_completude": 0,
        "aplicacao_normativa": 0,
        "aprovado": False,
        "erros_encontrados": [reason],
        "justificativa": reason,
        "evaluated_at": datetime.utcnow().isoformat(),
        "model": settings.MODEL_NAME,
    }
