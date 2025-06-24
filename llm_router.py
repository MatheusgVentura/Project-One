from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import time
from shared import buscar_contratos
import google.generativeai as genai

router = APIRouter()

# Configuração do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class QuestionRequest(BaseModel):
    question: str
    max_results: int = 50

class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Responde a perguntas sobre contratos usando o LLM com base nos resultados da busca semântica."""
    start_time = time.time()
    
    try:
        # Validação básica da pergunta
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia")
            
        print(f"[LLM] Recebida pergunta: '{request.question}' (max_results={request.max_results})")
        
        # 1. Busca direta no Pinecone usando a função buscar_documentos
        from pinecone_utils import buscar_documentos
        
        print(f"[LLM] Realizando busca semântica direta...")
        try:
            # Busca direta nos documentos
            documentos = buscar_documentos(request.question, request.max_results)
            
            # Verifica se há resultados
            if not documentos or len(documentos) == 0:
                print(f"[LLM] Nenhum documento relevante encontrado.")
                raise HTTPException(status_code=404, detail="Nenhum documento relevante encontrado.")
            
            print(f"[LLM] Encontrados {len(documentos)} documentos relevantes.")
        except Exception as e:
            print(f"[LLM] Erro na busca: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erro ao processar a consulta: {str(e)}")
        
        # 3. Prepara contexto para o LLM diretamente dos documentos encontrados
        context = "\n\n".join(
            f"[Documento {i+1} - {doc['arquivo']}]\n{doc['texto']}"
            for i, doc in enumerate(documentos)
        )

        # 4. Geração da resposta com o Gemini
        print(f"[LLM] Gerando resposta com o modelo Gemini...")
        
        try:
            # Configuração do modelo Gemini
            model = genai.GenerativeModel(model_name='gemini-2.0-flash')
            
            # Preparação do prompt
            prompt = f"""Você é um assistente especializado em contratos imobiliários com acesso a uma base de documentos. 
            Suas respostas devem ser:
            1. DETALHADAS - Forneça informações completas e abrangentes sobre o que foi perguntado.
            2. ESPECÍFICAS - Quando a pergunta for sobre pessoas, entidades ou cláusulas, inclua TODOS os detalhes disponíveis nos documentos.
            3. ESTRUTURADAS - Organize a resposta de forma clara, usando listas ou seções quando apropriado.
            4. BASEADAS EM EVIDÊNCIAS - Cite explicitamente de qual documento/contrato a informação foi extraída.
            5. Cite explicitamente codigos de barras, caso as informações sejam de boletos de cobrança.

            Documentos:
            {context}

            Pergunta: {request.question}"""

            # Geração da resposta
            response = model.generate_content(prompt)
            answer = response.text
            
        except Exception as e:
            print(f"[LLM] Erro ao gerar resposta: {str(e)}")
            raise HTTPException(status_code=500, detail="Erro ao gerar resposta. Tente novamente.")

        # 5. Prepara e retorna a resposta
        response = {
            "answer": answer,
            "sources": [{
                "filename": doc["arquivo"],
                "text": doc["texto"]
            } for doc in documentos]
        }
        
        return response
        
    except Exception as e:
        print(f"[LLM] Erro não tratado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")
