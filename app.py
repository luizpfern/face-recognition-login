from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), file: UploadFile = None):
    # 1. Salvar dados do usuário no banco
    # 2. Se tiver foto, extrair embedding facial e salvar
    return {"msg": "Usuário registrado com sucesso"}

@app.get("/login")
async def login():
    # 1. Validar credenciais
    # 2. Retornar token JWT
    print('entrei aqui')
    return {"msg": "Login realizado com sucesso", "token": "abc123"}

'''
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Servir app compilado a partir da pasta www
    return FileResponse(f"www/{full_path}" if full_path else "www/index.html")
'''

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
