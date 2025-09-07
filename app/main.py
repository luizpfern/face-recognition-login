from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# monta a pasta www
app.mount("/", StaticFiles(directory="www", html=True), name="www")

@app.post("/register")
async def register():
    return {"msg": f"Usuário teste registrado"}

@app.post("/login")
async def login():
    return {"msg": f"Login realizado para teste"}
