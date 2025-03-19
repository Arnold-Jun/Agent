from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
from typing import List, Optional

app = FastAPI()

model = BGEM3FlagModel('./bge-m3', use_fp16=True)

class SentenceRequest(BaseModel):
    inputs: List[str]
    truncate: Optional[int] = None  # 假设 truncate 是可选的

@app.post("/BAAI/bge-m3/embed", response_model=List[List[float]])
async def get_embedding(request: SentenceRequest):
    try:
        sentences = request.inputs

        if request.truncate:
            sentences = [s[:request.truncate] for s in sentences]
        embeddings = model.encode(sentences)['dense_vecs']
        return [embedding.tolist() for embedding in embeddings]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
