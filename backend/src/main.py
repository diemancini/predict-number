from fastapi import FastAPI
from predict_number import router

app = FastAPI()


app.include_router(router.router, prefix="/api/v1")
