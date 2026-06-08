import asyncio
import logging
import os
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from app.core.config import settings
from app.services.llm import get_chat_model_claude
from app.tools.rag import rag_results_context
from app.agent.workers import (
    compliance_agent_tool,
    audio_agent_tool
)
from app.services.memory import get_session_history
from app.services.validator import validate_response
from app.agent.evaluator import evaluate_response
from app.services.evaluation_store import save_evaluation

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    """Builds the system prompt with clear instructions for the clinical assistant."""
    return """Você é o Auxiliar Médico IA, um assistente especializado em apoiar consultas clínicas individuais em tempo real através da transcrição e auditoria de registros médicos.
Sua comunicação deve ser em português do Brasil, objetiva, profissional e orientada ao médico.

## Objetivos
1. **Estruturação Clínica**: Receber a transcrição da consulta e organizá-la de forma estruturada nos campos: ANAMNESE, CONDUTA, HIPÓTESE DIAGNÓSTICA, CID-10 e ORIENTAÇÕES.
2. **Auditoria de Qualidade**: Avaliar a clareza, completude e conformidade do prontuário com base nas normas do CFM e regulamentos de registro médico (RAG).
3. **Sugestão de Melhorias**: Apontar potenciais lacunas clínicas, pontos omissos e sugerir perguntas adicionais úteis ao médico, sem inventar fatos ou diagnosticar o paciente.

## Diretrizes de Auditoria (RAG)
- Para esclarecer critérios legais de preenchimento, normas de prontuário, regras do CFM ou manuais operacionais, você DEVE utilizar a ferramenta `compliance_agent_tool`.
- Use as informações retornadas do RAG para verificar se o registro do prontuário atende aos requisitos documentais mínimos estabelecidos.
- Sempre que houver necessidade de transcrever ou processar áudios para obter o texto clínico, você deve utilizar a `audio_agent_tool`.

## Regras de Segurança e Fidelidade Clínica
1. **Não Substitua o Médico**: Nunca apresente suas conclusões ou códigos CID-10 como diagnósticos definitivos ou prescrições aplicáveis. Apresente-os sempre como sugestões sujeitas à validação do profissional.
2. **Zero Alucinação Clínica**: Não invente dados clínicos (sintomas, dados vitais, exames ou medicamentos) que não foram expressamente citados na transcrição do áudio.
3. **Diferenciação Clara**: Sua resposta deve distinguir explicitamente:
   - O que foi relatado/dito durante a consulta (fatos originais).
   - O que foi inferido com baixo nível de confiança ou ambiguidade (alertando sobre limitações).
   - O que são sugestões de conformidade ou melhorias no prontuário.
4. **Sinalização de Ruído**: Se a transcrição estiver confusa, fragmentada ou com palavras incompreensíveis devido a ruídos, alerte o médico explicitamente sobre essas limitações.
"""


def extract_text_from_content(content: Any) -> str:
    """
    Extracts only user-facing text from LangChain/LangGraph/Anthropic content blocks.
    Ignores tool_use, input_json_delta, and any non-text structured content.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                # ignora tool_use, input_json_delta e outros blocos
                continue

            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))

        return "".join(parts)

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""

    content_type = getattr(content, "type", None)
    if content_type == "text":
        return str(getattr(content, "text", ""))

    return ""


async def _run_evaluation_background(
    user_id: str,
    message: str,
    final_response: str,
    rag_data: list,
    chat_history: str = "",
) -> None:
    """Executa a avaliação em background."""
    try:
        logger.info(
            f"Avaliador iniciado em background | user_id={user_id} "
            f"| queries_rag={len(rag_data)}"
        )
        evaluation = await evaluate_response(message, final_response, rag_data, chat_history)
        await save_evaluation(user_id, message, final_response, evaluation)
    except Exception:
        logger.exception("Erro no pipeline de avaliação em background")


async def run_agent(user_id: str, message: str, stream: bool = False):
    """
    Main agent entrypoint.
    - If stream=True, yields partial text chunks.
    - If stream=False, yields the final response once.
    """
    logger.info(f"Executando Agente Auxiliar Médico | user_id={user_id} | stream={stream}")

    if not message or not message.strip():
        yield "Por favor, digite uma mensagem."
        return

    # Zera os contextos de captura para esta execução
    rag_results_context.set([])

    env = os.getenv("RENDER", "development")
    tracing_metadata = {
        "user_id": user_id,
        "environment": env,
    }

    llm = get_chat_model_claude(model=settings.MODEL_ORCHESTRATOR, metadata=tracing_metadata)
    tools = [
        compliance_agent_tool,
        audio_agent_tool,
    ]
    system_prompt = _build_system_prompt()
    history = get_session_history(user_id)

    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        # Limita o contexto às últimas 10 conversas para economizar tokens lidos pelo LLM
        recent_messages = list(history.messages)[-10:]
        # OpenAI rejeita `name` em mensagens de role assistant/user/system.
        # Remove `name` de todas as mensagens do histórico.
        for m in recent_messages:
            if hasattr(m, 'name'):
                m.name = None
            m.additional_kwargs.pop('name', None)
        input_messages = recent_messages + [HumanMessage(content=message)]

        config = {
            "configurable": {"thread_id": user_id},
            "run_name": "Agente Auxiliar Medico",
            "metadata": tracing_metadata,
            "tags": [env, "agent"],
        }

        if stream:
            full_response = ""
            active_tools = 0

            try:
                async for event in agent.astream_events(
                    {"messages": input_messages},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_tool_start":
                        active_tools += 1
                        tool_name = event.get("name", "ferramenta")
                        if active_tools == 1:
                            logger.info(f"Executando ferramenta: {tool_name}")
                            yield f"\n[⚙️ Pensando: Acionando {tool_name}...]\n"
                        continue
                        
                    if kind == "on_tool_end":
                        active_tools -= 1
                        tool_name = event.get("name", "ferramenta")
                        if active_tools == 0:
                            yield f"\n[✅ {tool_name} finalizado]\n"
                        continue

                    if kind == "on_chat_model_stream":
                        if active_tools == 0:
                            chunk = event.get("data", {}).get("chunk")
                            if not chunk:
                                continue

                            text = extract_text_from_content(getattr(chunk, "content", None))
                            if text:
                                full_response += text
                                yield text

                final_response = validate_response(full_response).output if full_response else ""

                history.add_user_message(message)
                history.add_ai_message(final_response)

                # Dispara avaliação em background (sem bloquear o stream)
                rag_data = rag_results_context.get([])
                if final_response:
                    # Formata o histórico para o avaliador
                    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_messages])
                    asyncio.create_task(
                        _run_evaluation_background(user_id, message, final_response, rag_data, history_str)
                    )

            except asyncio.CancelledError:
                logger.warning("Streaming cancelado pelo cliente.")
                return

        else:
            result = await agent.ainvoke({"messages": input_messages}, config=config)

            messages = result.get("messages", [])
            if not messages:
                final_response = "Não foi possível gerar uma resposta."
            else:
                response_text = extract_text_from_content(messages[-1].content)
                validation = validate_response(response_text)
                final_response = validation.output

            # Retry automático em caso de erro técnico
            if final_response.startswith("Erro técnico:"):
                logger.warning("Resposta com erro técnico — refazendo consulta...")
                result = await agent.ainvoke({"messages": input_messages}, config=config)
                messages = result.get("messages", [])
                if messages:
                    response_text = extract_text_from_content(messages[-1].content)
                    validation = validate_response(response_text)
                    new_response = validation.output
                    if not new_response.startswith("Erro técnico:"):
                        final_response = new_response
                        logger.info("Retry bem-sucedido")

            history.add_user_message(message)
            history.add_ai_message(final_response)

            rag_data = rag_results_context.get([])
            if final_response:
                history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_messages])
                asyncio.create_task(
                    _run_evaluation_background(user_id, message, final_response, rag_data, history_str)
                )

            yield final_response

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"