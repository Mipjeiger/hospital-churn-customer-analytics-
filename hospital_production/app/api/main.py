import logging
from fastapi import FastAPI
from api.routes import llm
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Build FastAPI app
app = FastAPI(title="Hospital Operations API", version="1.0.0")

# Include API routers
app.include_router(llm.router)

# Endpoint helper
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Hospital Operations API is running."}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)