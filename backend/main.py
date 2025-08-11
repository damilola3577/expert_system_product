from fastapi import FastAPI, Form
from pydantic import BaseModel
import numpy as np
from optimizer import run_ga
from rag import KnowledgeIndex, rag_query

app = FastAPI()
index = KnowledgeIndex(docs_dir="knowledge")

class OptimizeRequest(BaseModel):
    tariff: list
    desired: float = 21.0
    comfort_weight: float = 1.0
    outdoor_temp: float = 10.0
    pop_size: int = 50
    gens: int = 120

@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    tariff = np.array(req.tariff, dtype=float)
    res = run_ga(tariff,
                 pop_size=req.pop_size,
                 gens=req.gens,
                 desired=req.desired,
                 comfort_weight=req.comfort_weight,
                 outdoor_temp=req.outdoor_temp)
    return res

@app.post("/rag_query")
async def query(q: str = Form(...)):
    answer = rag_query(q, index, k=3)
    return {"answer": answer}
