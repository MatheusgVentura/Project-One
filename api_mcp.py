from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from controller.controller import Controller
from policy.policy import Policy

app = FastAPI(title="API MCP - RAG com Memory, Controller, Policy")

# Instancia as camadas MCP
controller = Controller()
policy = Policy()

class PerguntaRequest(BaseModel):
    pergunta: str
    max_results: int = 5

class RespostaMCP(BaseModel):
    answer: str
    sources: list

@app.post("/mcp/ask", response_model=RespostaMCP)
def mcp_ask(request: PerguntaRequest):
    """
    Endpoint MCP: recebe uma pergunta, aplica políticas, consulta memória e gera resposta com LLM.
    """
    # 1. Valida a pergunta
    if not policy.validar_pergunta(request.pergunta):
        raise HTTPException(status_code=400, detail="Pergunta inválida ou muito curta.")
    # 2. Busca documentos relevantes
    documentos = controller.memory.buscar_documentos(request.pergunta, top_k=request.max_results)
    documentos_filtrados = policy.aplicar_politicas(request.pergunta, documentos)
    if not documentos_filtrados:
        raise HTTPException(status_code=404, detail="Nenhum documento relevante encontrado.")
    # 3. Monta contexto e gera resposta
    contexto = "\n\n".join(
        f"[Documento {i+1} - {doc['arquivo']}]:\n{doc['texto']}"
        for i, doc in enumerate(documentos_filtrados)
    )
    prompt = f"""Você é um assistente especializado em contratos imobiliários.\n"""
    prompt += "Responda de forma detalhada, cite os documentos usados e organize a resposta.\n"
    prompt += f"Documentos:\n{contexto}\n\nPergunta: {request.pergunta}"
    try:
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyBfyEYsdM5XiuXNFlX2tTpj2IVN5wnHo00")
        print(genai.list_models())
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {str(e)}")
    return {
        "answer": answer,
        "sources": [
            {"filename": doc["arquivo"], "text": doc["texto"]}
            for doc in documentos_filtrados
        ]
    } 