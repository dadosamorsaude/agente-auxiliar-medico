import sys
import asyncio
import os
from dotenv import load_dotenv

# Reconfigura stdout para utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Adiciona o diretório atual ao sys.path para permitir imports do app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from app.agent.orchestrator import run_agent
from app.agent.evaluator import evaluate_response
from app.services.evaluation_store import save_evaluation

async def test_flow():
    print("=== TESTANDO FLUXO AUXILIAR MÉDICO ===")
    user_id = "medico_teste_123"
    
    # Pergunta contendo uma transcrição simulada
    pergunta = (
        "Paciente João da Silva, 45 anos, queixa-se de cefaleia holocraniana há 3 dias. "
        "Exame físico geral sem alterações. Conduta: Receitado dipirona 500mg de 6 em 6 horas se dor, "
        "e orientado a retornar se houver piora. Hipótese diagnóstica: cefaleia tensional. CID-10: G44.2."
    )
    
    print(f"\n1. Executando o agente com a transcrição simulada...\n")
    full_response = ""
    async for chunk in run_agent(user_id, pergunta, stream=False):
        if chunk:
            full_response += chunk
            print(chunk, end="", flush=True)
    print()
    
    print("\n" + "="*50)
    print("2. Executando o avaliador (LLM-as-Judge)...")
    print("="*50)
    
    # O avaliador do auxiliar-medico não usa mais dados brutos do Athena
    eval_res = await evaluate_response(
        user_question=pergunta,
        agent_response=full_response,
        rag_context=None
    )
    
    print(f"Resultado do Judge:")
    print(f"  Score Geral: {eval_res.get('score')}/100")
    print(f"  Fidelidade Clínica: {eval_res.get('fidelidade_clinica')}/40")
    print(f"  Estruturação & Completude: {eval_res.get('estruturacao_completude')}/30")
    print(f"  Aplicação Normativa: {eval_res.get('aplicacao_normativa')}/30")
    print(f"  Aprovado: {eval_res.get('aprovado')}")
    print(f"  Erros encontrados: {eval_res.get('erros_encontrados')}")
    print(f"  Justificativa: {eval_res.get('justificativa')}")
    
    print("\n3. Salvando avaliação no banco de dados...")
    await save_evaluation(
        user_id=user_id,
        question=pergunta,
        response=full_response,
        evaluation=eval_res
    )
    print("Avaliação salva com sucesso!")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_flow())
