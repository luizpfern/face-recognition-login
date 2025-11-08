from typing import List, Dict

# Repositório em memória simples (substituir por DB quando precisar)
class InMemoryUserRepo:
    def __init__(self):
        self._store: Dict[str, List[List[float]]] = {}

    def append_embeddings(self, username: str, embeddings: List[List[float]]) -> int:
        if username not in self._store:
            self._store[username] = []
        self._store[username].extend(embeddings)
        return len(self._store[username])

    def load_embeddings(self, username: str) -> List[List[float]]:
        return self._store.get(username, [])

    def user_exists(self, username: str) -> bool:
        """Retorna True se o usuário já estiver presente no repositório em memória.

        A existência é verificada pela presença da chave no dicionário interno.
        """
        return username in self._store