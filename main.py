from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from fastapi.responses import FileResponse
from fastapi.exception_handlers import http_exception_handler
from app.services import (
    decode_image_bytes,
    detect_and_validate,
    align_face,
    preprocess_face,
    get_embedding,
    compare_embeddings,
)
from app.repository import InMemoryUserRepo

app = FastAPI(title="FaceAuth API", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# repositório simples em memória (injeção direta para simplicidade)
user_repo = InMemoryUserRepo()


@app.post("/register")
async def register(
    username: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not username:
        raise HTTPException(status_code=400, detail="username_required")
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="images_required")
    
    user_repo.print_store()  # Debug: imprime o estado atual do repositório


    # Verifica se usuário já existe para evitar registros duplicados
    if user_repo.user_exists(username):
        return {"success": False, "reason": "user_already_exists"}

    embeddings = []
    for f in images:
        data = await f.read()
        img = decode_image_bytes(data)
        ok, reason = detect_and_validate(img)
        if not ok:
            # early return com motivo para UX
            return {"success": False, "reason": reason}
        aligned = align_face(img)
        face = preprocess_face(aligned)
        enc = get_embedding(face)
        if enc is None:
            return {"success": False, "reason": "encoding_failed"}
        embeddings.append(enc.tolist())

    stored = user_repo.append_embeddings(username, embeddings)
    return {"success": True, "stored_embeddings": stored}
    

@app.post("/login")
async def login(
    username: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not username:
        raise HTTPException(status_code=400, detail="username_required")
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="images_required")
    
    user_repo.print_store()  # Debug: imprime o estado atual do repositório

    known = user_repo.load_embeddings(username)
    if not known:
        return {"authenticated": False, "reason": "user_not_found"}
    
    candidate_embeddings = []
    for f in images:
        data = await f.read()
        img = decode_image_bytes(data)
        ok, reason = detect_and_validate(img)
        if not ok:
            return {"authenticated": False, "reason": reason}
        aligned = align_face(img)
        face = preprocess_face(aligned)
        enc = get_embedding(face)
        if enc is None:
            return {"authenticated": False, "reason": "encoding_failed"}
        candidate_embeddings.append(enc)

    result = compare_embeddings(known, candidate_embeddings)
    return result

# Tratamento customizado para erros 404
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        # after reorganizing HTML files into /www/pages we need to serve the 404 from there
        return FileResponse("www/pages/notfound.html", status_code=404, media_type="text/html")
    return await http_exception_handler(request, exc)


# Serve the app entrypoint at root so uvicorn/main dev server returns the expected page
@app.get("/", include_in_schema=False)
async def root_index():
    return FileResponse("www/pages/index.html", media_type="text/html")

# Retorna a interface web (colocado depois das rotas para não interceptar requisições POST)
app.mount("/", StaticFiles(directory="www", html=True), name="www")