from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import router as api_router

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI()
app.include_router(api_router)
app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")


@app.get("/")
async def home():
    return FileResponse(FRONTEND / "home.html")


@app.get("/settings")
async def settings():
    return FileResponse(FRONTEND / "settings.html")


@app.get("/test")
async def test():
    return FileResponse(FRONTEND / "test.html")
