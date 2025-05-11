import os
import sys
import uuid
import logging
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# Ensure data directory exists
project_root = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(project_root, '..')))

from backend.vectordb.chroma_db import ChromaDBManager
from backend.llm.meta_llama import ask_meta_llama_rag, check_document_relevance
from backend.utils.chat_memory import get_session_memory, reset_session_memory

app = FastAPI()
vectordb = ChromaDBManager()

class QueryRequest(BaseModel):
    query: str
    document_id: str
    session_id: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    document_id = str(uuid.uuid4())[:8]
    filename = f"{document_id}_{file.filename}"
    file_path = os.path.join(data_dir, filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        vectordb.add_pdf(file_path, document_id=document_id)
        chunks = vectordb.get_documents(document_id=document_id)
        logger.info(f"Indexed {len(chunks)} chunks for document_id={document_id}")
        if not chunks:
            raise HTTPException(status_code=500, detail="PDF indexing failed.")
        return {"message": f"PDF '{file.filename}' uploaded successfully!",
                "document_id": document_id,
                "indexed_documents": len(chunks)}
    except Exception as e:
        logger.exception("Error in upload_pdf")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
def get_documents(document_id: str):
    docs = vectordb.get_documents(document_id=document_id)
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for this document_id")
    return {"document_id": document_id, "documents": docs}

@app.post("/rag")
def rag_query(request: QueryRequest):
    query = request.query
    document_id = request.document_id
    session_id = request.session_id
    if not session_id or not document_id:
        raise HTTPException(status_code=400, detail="Both session_id and document_id are required.")

    memory = get_session_memory(session_id)
    prev_history = memory.load_memory_variables({}).get("history", "")

    # Scoped and global retrieval inside hybrid_query now handles errors
    docs, metas = vectordb.hybrid_query(query, document_id=document_id, top_k=5)
    used_filter = True if docs else False
    logger.info(f"Retrieved {len(docs)} chunks (used_filter={used_filter}) for DocID={document_id}")

    # Build context from filtered docs
    filtered = check_document_relevance(query, docs)
    context = "\n".join(
        f"{doc}\n[Keywords: {', '.join(md.get('keywords', []))}]" for doc, md in zip(filtered, metas)
    )

    # Generate LLM response
    response = ask_meta_llama_rag(query, prev_history, context)
    memory.save_context({"input": query}, {"output": response})

    return {
        "query": query,
        "response": response,
        "session_id": session_id,
        "chat_history": memory.load_memory_variables({}).get("history", ""),
        "retrieved_documents": len(filtered),
        "used_document_filter": used_filter
    }

@app.post("/reset_chat/")
def reset_chat(request: BaseModel = QueryRequest):
    return reset_session_memory(request.session_id)

if __name__ == "__main__":
    import uvicorn
    module = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=8000, reload=False)

