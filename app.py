import uvicorn
from fastapi import FastAPI
from server.routes import router

app = FastAPI(title="IC LLM RAG Pipeline Dashboard")

# Include all API routes and the main page route
app.include_router(router)

if __name__ == "__main__":
    print("\nIniciando servidor do painel do RAG em http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
