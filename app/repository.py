from typing import List, Dict
import json
import os
from pathlib import Path

# Repositório com persistência em arquivo JSON
class InMemoryUserRepo:
    def __init__(self, storage_file: str = "users_data.json"):
        """
        Inicializa o repositório com persistência em arquivo JSON.
        
        Args:
            storage_file: Caminho do arquivo JSON para persistência dos dados
        """
        self.storage_file = storage_file
        self._store: Dict[str, List[List[float]]] = {}
        self._load_from_file()

    def _load_from_file(self):
        """Carrega dados do arquivo JSON se existir."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self._store = json.load(f)
                self.print_store()
            except Exception as e:
                self._store = {}
        else:
            self._store = {}

    def _save_to_file(self):
        """Salva dados no arquivo JSON."""
        try:
            # Garante que o diretório existe
            Path(self.storage_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self._store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar dados em {self.storage_file}: {e}")

    def append_embeddings(self, username: str, embeddings: List[List[float]]) -> int:
        if username not in self._store:
            self._store[username] = []
        self._store[username].extend(embeddings)
        print(f"Embeddings stored for {username}: {len(self._store[username])}")
        
        # Persiste os dados após adicionar
        self._save_to_file()
        
        return len(self._store[username])

    def load_embeddings(self, username: str) -> List[List[float]]:
        self.print_store()  # Debug: imprime o estado atual do repositório
        return self._store.get(username, [])

    def user_exists(self, username: str) -> bool:
        """Retorna True se o usuário já estiver presente no repositório em memória.

        A existência é verificada pela presença da chave no dicionário interno.
        """
        return username in self._store
    
    def print_store(self):
        """Função auxiliar para debug: imprime o conteúdo do repositório."""
        if not self._store:
            print("📋 Repositório vazio - nenhum usuário cadastrado")
        else:
            print(f"📋 Repositório contém {len(self._store)} usuário(s):")
            for user, embeddings in self._store.items():
                print(f"   - User: {user}, Embeddings count: {len(embeddings)}")