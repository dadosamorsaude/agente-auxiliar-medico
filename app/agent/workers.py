import logging
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from app.services.llm import get_chat_model_openai
from app.core.config import settings
from app.tools.rag import search_medical_compliance_tool, search_sop_tool
from app.tools.transcription import transcribe_audio_tool

logger = logging.getLogger(__name__)

# Agente de Compliance e Manuais
@tool("compliance_agent_tool")
async def compliance_agent_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em Compliance Médico, Regras CFM e POPs Internos.
    Use este agente para tirar dúvidas sobre como avaliar a qualidade, regras do CFM ou manuais operacionais.
    """
    logger.info("Executando Compliance Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_COMPLIANCE)
    agent = create_react_agent(
        model=llm, 
        tools=[search_medical_compliance_tool, search_sop_tool],
        prompt="You are an expert in medical norms and auditing. Use your tools to search for information and pass it on. ALWAYS respond in Brazilian Portuguese."
    )
    child_config = {**config, "run_name": "Compliance RAG Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content

# Agente de Áudio / Transcrição
@tool("audio_agent_tool")
async def audio_agent_tool(query: str, config: RunnableConfig) -> str:
    """
    Agente Especialista em Processamento de Áudio Clínico.
    Use este agente para transcrever ditados médicos.
    """
    logger.info("Executando Audio Agent...")
    llm = get_chat_model_openai(model=settings.MODEL_AUDIO)
    agent = create_react_agent(
        model=llm, 
        tools=[transcribe_audio_tool],
        prompt="You transcribe and structure medical dictations. ALWAYS respond in Brazilian Portuguese."
    )
    child_config = {**config, "run_name": "Audio Transcription Agent"}
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=child_config)
    return result["messages"][-1].content