import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import router as api_router

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": str(exc)})


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
