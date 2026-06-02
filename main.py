from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import measurements

app = FastAPI(
    title="Body Measurement API",
    description="AI-powered body measurement estimation from photos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(measurements.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
