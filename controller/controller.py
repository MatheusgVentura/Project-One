"""
Módulo Controller
Responsável por orquestrar o fluxo de perguntas e respostas:
- Recebe a pergunta do usuário
- Consulta a memória
- Aplica políticas
- Monta o contexto e chama o LLM
"""

from memory.pinecone_memory import PineconeMemory
import google.generativeai as genai

class Controller:
    """
    Classe responsável por orquestrar o fluxo RAG-MCP:
    - Recebe a pergunta
    - Consulta a memória
    - (Futuramente) Aplica políticas
    - Monta o contexto e chama o LLM
    """
    def __init__(self):
        self.memory = PineconeMemory()
        # Configura o Gemini (pode ser customizado)
        # A chave já é configurada na PineconeMemory

    def responder_pergunta(self, pergunta: str, max_results: int = 5) -> dict:
        """
        Recebe uma pergunta, busca contexto relevante e gera uma resposta com o LLM.
        """
        if not pergunta or not pergunta.strip():
            return {"error": "Pergunta vazia."}
        # 1. Busca documentos relevantes na memória
        documentos = self.memory.buscar_documentos(pergunta, top_k=max_results)
        if not documentos:
            return {"error": "Nenhum documento relevante encontrado."}
        # 2. Monta o contexto para o LLM
        contexto = "\n\n".join(
            f"[Documento {i+1} - {doc['arquivo']}]:\n{doc['texto']}"
            for i, doc in enumerate(documentos)
        )
        # 3. Prepara o prompt
        prompt = f"""Você é um assistente especializado em contratos imobiliários.\n"""
        prompt += "Responda de forma detalhada, cite os documentos usados e organize a resposta.\n"
        prompt += f"Documentos:\n{contexto}\n\nPergunta: {pergunta}"
        # 4. Gera a resposta com Gemini
        try:
            model = genai.GenerativeModel(model_name='gemini-2.0-flash')
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            return {"error": f"Erro ao gerar resposta: {str(e)}"}
        # 5. Retorna resposta e fontes
        return {
            "answer": answer,
            "sources": [
                {"filename": doc["arquivo"], "text": doc["texto"]}
                for doc in documentos
            ]
        } 