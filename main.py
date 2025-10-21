from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from fastapi.responses import FileResponse
from fastapi.exception_handlers import http_exception_handler
app = FastAPI(title="FaceAuth API", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/register")
async def register(
    username: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not username:
        raise HTTPException(status_code=400, detail="username_required")
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="images_required")

    # Logica de registro ficaria aqui
    return {"msg": f"Registro realizado para {username} com {len(images)} imagens"}
    

@app.post("/login")
async def login(
    username: str = Form(...),
    image: UploadFile = File(...)
):
    if not username:
        raise HTTPException(status_code=400, detail="username_required")
    if not image:
        raise HTTPException(status_code=400, detail="image_required")
    
    # Logica de login ficaria aqui
    return {"msg": f"Login realizado para {username}"}

# Tratamento customizado para erros 404
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return FileResponse("www/notfound.html", status_code=404, media_type="text/html")
    return await http_exception_handler(request, exc)

# Retorna a interface web (colocado depois das rotas para não interceptar requisições POST)
app.mount("/", StaticFiles(directory="www", html=True), name="www")