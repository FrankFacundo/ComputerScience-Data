from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from .service import predict

ROOT_PATH = '/api/atp'

class PredictPayload(BaseModel):
    player1: str
    player2: str

app = FastAPI(root_path = ROOT_PATH)

@app.get('/')
async def health_check():
    return Response(content='OK', status_code=200)

@app.post('/predict/')
async def handlerpredict(payload: PredictPayload):
    prediction = predict(payload)
    if prediction:
        return "The winner is the player 2. "
    return "The winner is the player 1. "
