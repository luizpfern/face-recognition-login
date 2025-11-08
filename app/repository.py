from typing import List, Dict
import pickle
import os

# try imports that may not be installed in prod; fall back gracefully
try:
    import psutil
except Exception:
    psutil = None

try:
    from pympler import asizeof
except Exception:
    asizeof = None

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

    def _log_process_mem(self, label: str = "") -> None:
        """Imprime PID e RSS do processo atual; usa psutil se disponível."""
        pid = os.getpid()
        if psutil:
            try:
                p = psutil.Process(pid)
                rss_mb = p.memory_info().rss / 1024 / 1024
                print(f"[repo][mem] {label} PID={pid} RSS={rss_mb:.1f} MB")
                return
            except Exception:
                pass

        # Fallback: tentar ler /proc/self/status (Linux/Heroku)
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        # VmRSS em kB
                        kb = float(parts[1])
                        print(f"[repo][mem] {label} PID={pid} RSS={kb/1024:.1f} MB (from /proc)")
                        return
        except Exception:
            pass

        print(f"[repo][mem] {label} PID={pid} RSS=unknown (install psutil for exact RSS)")

    def append_embeddings(self, username: str, embeddings: List[List[float]]) -> int:
        # log antes
        self._log_process_mem(label="before append")

        if username not in self._store:
            self._store[username] = []
        self._store[username].extend(embeddings)

        # Print dos tamanhos em KB para debug/monitoramento
        print(f"[repo] append_embeddings: adicionado {len(embeddings)} embeddings para usuário '{username}'")

        # log deep-size se disponível
        if asizeof:
            try:
                deep_kb = asizeof.asizeof(self._store.get(username, [])) / 1024
                print(f"[repo][deep] tamanho profundo para '{username}': {deep_kb:.2f} KB")
            except Exception:
                pass

        self._print_sizes_for_user(username)

        # log depois
        self._log_process_mem(label="after append")

        return len(self._store[username])

    def load_embeddings(self, username: str) -> List[List[float]]:
        # log antes do load
        self._log_process_mem(label="load start")

        entries = self._store.get(username, [])
        # Ao carregar, também imprimimos os tamanhos para inspeção
        print(f"[repo] load_embeddings: carregando {len(entries)} embeddings para usuário '{username}'")
        self._print_sizes_for_user(username)

        # log depois do load
        self._log_process_mem(label="load end")

        return entries