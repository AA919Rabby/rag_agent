from fastapi import FastAPI
from app.config import settings
from app.api.v1.endpoints import chat


app=FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(chat.router,prefix="/api/v1",tags=["PDF RAG System"])

@app.get("/",tags=["Health Check"])
def health_check():
    return {"status":"ok"}