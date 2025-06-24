import os
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Carrega as variáveis de ambiente
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(diretorio_atual, '.env')
load_dotenv(dotenv_path=env_path)

# Configurações do Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "brito-ai")
EMBEDDING_DIM = 768

# Configurações do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def inicializar_pinecone():
    """Inicializa a conexão com o Pinecone e retorna o índice."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY não encontrada no arquivo .env")
    
    if not PINECONE_HOST:
        raise ValueError("PINECONE_HOST não encontrado no arquivo .env")
    
    try:
        # Inicializa o cliente Pinecone com a API V2
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Conecta ao índice
        index = pc.Index(INDEX_NAME, host=PINECONE_HOST)
        
        # Verifica se o índice está acessível obtendo suas estatísticas
        stats = index.describe_index_stats()
        print(f"Conexão com o índice '{INDEX_NAME}' estabelecida com sucesso!")
        print(f"Total de vetores no índice: {stats.get('total_vector_count', 0)}")
        
        return index
    except Exception as e:
        print(f"Erro ao conectar ao Pinecone: {e}")
        raise

def gerar_embedding(texto):
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
        print(f"Erro ao gerar embedding: {e}")
        raise

def processar_e_indexar_documento(texto, metadata, id=None, index=None):
    """
    Processa um documento e o indexa no Pinecone.
    
    Args:
        texto: Texto do documento
        metadata: Metadados do documento (dict)
        id: ID opcional do documento
        index: Índice Pinecone (opcional, será inicializado se None)
    
    Returns:
        ID do documento indexado
    """
    try:
        # Gera o embedding usando o modelo do Gemini
        embedding = gerar_embedding(texto)
        
        # Gera um ID se não fornecido
        if id is None:
            id = str(int(time.time() * 1000))  # Timestamp como ID
        
        # Usa o índice fornecido ou inicializa um novo
        if index is None:
            index = inicializar_pinecone()
        
        # Upsert no Pinecone
        index.upsert(vectors=[(id, embedding, metadata)])
        
        return id
    except Exception as e:
        print(f"Erro ao indexar documento: {e}")
        raise

def buscar_documentos(query, top_k=5):
    """
    Realiza uma busca semântica no Pinecone.
    
    Args:
        query: Texto da consulta
        top_k: Número máximo de resultados
    
    Returns:
        Lista de documentos mais relevantes
    """
    if not query or not query.strip():
        print("Erro: consulta vazia enviada para buscar_documentos")
        return []
        
    try:
        # Limita o tamanho da consulta para evitar problemas
        query_processada = query.strip()
        if len(query_processada) > 1000:
            query_processada = query_processada[:1000]
            print(f"Aviso: consulta truncada para 1000 caracteres. Original: '{query[:30]}...'")
        
        # Verifica se é uma consulta sobre valores ou uma pessoa específica
        consulta_sobre_valor = any(termo in query_processada.lower() for termo in ["valor", "preço", "custo", "aluguel", "taxa", "multa", "reais", "r$", "pagamento"])
        consulta_sobre_pessoa = any(nome in query_processada.lower() for nome in ["eduardo", "rocha", "fontenele", "gabriela", "bruno", "ana"])
        
        # Enriquece a consulta para melhorar os resultados
        if consulta_sobre_valor:
            query_processada = f"{query_processada} valor aluguel preço pagamento R$"
            print(f"Consulta sobre valores detectada, consulta enriquecida: '{query_processada[:50]}...'")
        elif consulta_sobre_pessoa:
            query_processada = f"{query_processada} nome cpf rg identificação contratante locatário inquilino"
            print(f"Consulta sobre pessoa detectada, consulta enriquecida: '{query_processada[:50]}...'")
            
        # Gera o embedding da consulta usando o modelo do Gemini
        try:
            query_embedding = gerar_embedding(query_processada)
        except Exception as e:
            print(f"Erro ao gerar embedding para a consulta: {e}")
            raise ValueError(f"Não foi possível gerar embedding para a consulta: {str(e)}")
        
        # Inicializa o Pinecone e conecta ao índice
        try:
            index = inicializar_pinecone()
            if not index:
                raise ConnectionError("Não foi possível conectar ao Pinecone")
        except Exception as e:
            print(f"Erro ao conectar ao Pinecone: {e}")
            raise ConnectionError(f"Falha na conexão com o Pinecone: {str(e)}")
        
        # Realiza a busca
        try:
            # Realizar a busca sem filtros para garantir resultados mais abrangentes
            resultados = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not resultados or not hasattr(resultados, 'matches') or not resultados.matches:
                print(f"Aviso: nenhum resultado encontrado para a consulta '{query_processada[:30]}...'")
                return []
                
        except Exception as e:
            print(f"Erro ao realizar busca no Pinecone: {e}")
            raise ValueError(f"Falha na busca vetorial: {str(e)}")
        
        # Formata os resultados
        documentos = []
        for match in resultados.matches:
            documentos.append({
                "arquivo": match.metadata.get("arquivo", ""),
                "texto": match.metadata.get("texto", ""),
                "score": match.score
            })
            
        return documentos
        
    except Exception as e:
        print(f"Erro na busca de documentos: {e}")
        raise

def listar_todos_documentos(limit=100):
    """
    Lista todos os documentos no índice.
    
    Args:
        limit: Número máximo de documentos a retornar
    
    Returns:
        Lista de documentos e total
    """
    try:
        # Inicializa o Pinecone e conecta ao índice
        index = inicializar_pinecone()
        
        # Obtém estatísticas do índice
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        
        if total == 0:
            return [], 0
        
        # Para listar documentos, precisamos fazer uma consulta genérica
        # já que a nova API do Pinecone não tem um método direto para listar todos os vetores
        try:
            # Criamos um vetor de zeros com a dimensão correta (768 para text-embedding-3-small)
            dummy_vector = [0.0] * EMBEDDING_DIM
            
            # Fazemos uma consulta com um limite alto
            resultados = index.query(
                vector=dummy_vector,
                top_k=min(limit, 10000),  # Limitado a 10000 pela API
                include_metadata=True
            )
            
            documentos = []
            for match in resultados.matches:
                documentos.append({
                    "id": match.id,
                    "arquivo": match.metadata.get("arquivo", ""),
                    "texto": match.metadata.get("texto", "")
                })
            
            return documentos, total
        
        except Exception as e:
            print(f"Erro ao listar documentos: {e}")
            return [], total
    except Exception as e:
        print(f"Erro ao inicializar Pinecone: {e}")
        return [], 0
