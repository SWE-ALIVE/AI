from fastapi import FastAPI
from app.api.routes import router
import os

app = FastAPI()

app.include_router(router, prefix="/api")

URL = os.getenv("URL", "http://default-url.com")


@app.get("/")
def root():
    return {"message": f"{URL}/docs 로 이동하여 API 문서를 확인하세요."}
