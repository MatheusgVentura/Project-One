"""
Módulo Policy
Define regras e políticas para o fluxo de perguntas e respostas:
- Restrições de acesso
- Lógica de decisão
- Filtros e validações
"""

class Policy:
    """
    Classe responsável por aplicar regras e políticas no fluxo MCP.
    """
    def __init__(self):
        # Aqui podem ser carregadas regras dinâmicas, configs, etc.
        pass

    def validar_pergunta(self, pergunta: str) -> bool:
        """
        Exemplo de validação: pergunta não pode ser vazia ou muito curta.
        """
        if not pergunta or not pergunta.strip():
            return False
        if len(pergunta.strip()) < 5:
            return False
        return True

    def filtrar_resultados(self, documentos: list) -> list:
        """
        Exemplo de filtro: remove documentos com score muito baixo.
        """
        return [doc for doc in documentos if doc.get('score', 0) > 0.2]

    def aplicar_politicas(self, pergunta: str, documentos: list) -> list:
        """
        Método geral para aplicar todas as políticas necessárias.
        """
        docs_filtrados = self.filtrar_resultados(documentos)
        # Outras políticas podem ser aplicadas aqui
        return docs_filtrados 