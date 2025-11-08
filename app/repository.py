from typing import List, Dict
import pickle

# Repositório em memória simples (substituir por DB quando precisar)
class InMemoryUserRepo:
    def __init__(self):
        self._store: Dict[str, List[List[float]]] = {}

    def _bytes_size(self, obj) -> int:
        """Retorna o tamanho em bytes de um objeto serializado via pickle."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # fallback simples: usar str()
            return len(pickle.dumps(str(obj)))

    def _print_sizes_for_user(self, username: str) -> None:
        """Imprime o tamanho em KB do username e de cada embedding associado."""
        entries = self._store.get(username, [])
        if not entries:
            print(f"[repo] usuário '{username}' sem embeddings armazenados.")
            return

        username_bytes = self._bytes_size(username)
        print(f"[repo] usuário '{username}' - tamanho nome: {username_bytes/1024:.2f} KB")

        total_bytes = username_bytes
        for idx, emb in enumerate(entries, start=1):
            emb_bytes = self._bytes_size(emb)
            total_bytes += emb_bytes
            print(f"[repo]   registro {idx}: embedding tamanho: {emb_bytes/1024:.2f} KB")

        print(f"[repo]   total (nome + embeddings) para '{username}': {total_bytes/1024:.2f} KB")

    def append_embeddings(self, username: str, embeddings: List[List[float]]) -> int:
        if username not in self._store:
            self._store[username] = []
        self._store[username].extend(embeddings)

        # Print dos tamanhos em KB para debug/monitoramento
        print(f"[repo] append_embeddings: adicionado {len(embeddings)} embeddings para usuário '{username}'")
        self._print_sizes_for_user(username)

        return len(self._store[username])

    def load_embeddings(self, username: str) -> List[List[float]]:
        entries = self._store.get(username, [])
        # Ao carregar, também imprimimos os tamanhos para inspeção
        print(f"[repo] load_embeddings: carregando {len(entries)} embeddings para usuário '{username}'")
        self._print_sizes_for_user(username)
        return entries