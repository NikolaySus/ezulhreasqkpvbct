from fastapi import FastAPI

app = FastAPI(title="MVP Backend")


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "backend", "status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
async def api_health() -> dict[str, str]:
    return {"status": "ok", "service": "local-backend"}


@app.get("/api/hello")
async def api_hello() -> dict[str, str]:
    return {"message": "Hello from the local FastAPI backend"}
