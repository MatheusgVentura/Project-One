"""
Módulo de Memória (Memory)
Responsável por interagir com o armazenamento vetorial (Pinecone):
- Salvar embeddings
- Buscar contexto relevante
- Gerenciar histórico, se necessário
"""

import os
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv
import time

class PineconeMemory:
    """
    Classe responsável por interagir com o Pinecone para salvar e buscar embeddings.
    """
    def __init__(self):
        # Carrega variáveis de ambiente
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(diretorio_atual, '..', '.env')
        load_dotenv(dotenv_path=env_path)

        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_HOST = os.getenv("PINECONE_HOST")
        self.INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "brito-ai")
        self.EMBEDDING_DIM = 768

        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.GEMINI_API_KEY)

        self.index = self._inicializar_pinecone()

    def _inicializar_pinecone(self):
        if not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY não encontrada no arquivo .env")
        if not self.PINECONE_HOST:
            raise ValueError("PINECONE_HOST não encontrado no arquivo .env")
        try:
            pc = Pinecone(api_key=self.PINECONE_API_KEY)
            index = pc.Index(self.INDEX_NAME, host=self.PINECONE_HOST)
            stats = index.describe_index_stats()
            print(f"[Memory] Conexão com o índice '{self.INDEX_NAME}' estabelecida com sucesso!")
            print(f"[Memory] Total de vetores no índice: {stats.get('total_vector_count', 0)}")
            return index
        except Exception as e:
            print(f"[Memory] Erro ao conectar ao Pinecone: {e}")
            raise

    def gerar_embedding(self, texto):
        """
        Gera um embedding usando o modelo do Gemini.
        """
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=texto,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            print(f"[Memory] Erro ao gerar embedding: {e}")
            raise

    def salvar_documento(self, texto, metadata, id=None):
        """
        Salva (upsert) um documento no Pinecone.
        """
        try:
            embedding = self.gerar_embedding(texto)
            if id is None:
                id = str(int(time.time() * 1000))
            self.index.upsert(vectors=[(id, embedding, metadata)])
            print(f"[Memory] Documento salvo com id: {id}")
            return id
        except Exception as e:
            print(f"[Memory] Erro ao salvar documento: {e}")
            raise

    def buscar_documentos(self, query, top_k=5):
        """
        Busca documentos relevantes no Pinecone a partir de uma consulta textual.
        """
        if not query or not query.strip():
            print("[Memory] Consulta vazia enviada para buscar_documentos")
            return []
        try:
            query_processada = query.strip()
            if len(query_processada) > 1000:
                query_processada = query_processada[:1000]
                print(f"[Memory] Aviso: consulta truncada para 1000 caracteres.")
            query_embedding = self.gerar_embedding(query_processada)
            resultados = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            if not resultados or not hasattr(resultados, 'matches') or not resultados.matches:
                print(f"[Memory] Nenhum resultado encontrado para a consulta.")
                return []
            documentos = []
            for match in resultados.matches:
                documentos.append({
                    "arquivo": match.metadata.get("arquivo", ""),
                    "texto": match.metadata.get("texto", ""),
                    "score": match.score
                })
            return documentos
        except Exception as e:
            print(f"[Memory] Erro na busca de documentos: {e}")
            raise 