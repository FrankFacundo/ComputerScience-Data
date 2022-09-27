from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from services.services import translate

class PredictPayload(BaseModel):
    text: str

app = FastAPI()

@app.get('/')
async def health_check():
    return Response(content='OK', status_code=200)

@app.post('/translate')
async def handler_predict(payload: PredictPayload):
    result = translate(payload)
    return result
