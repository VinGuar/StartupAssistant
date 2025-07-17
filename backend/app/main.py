from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import startup_assistant

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(startup_assistant.router, prefix="/api") 