from fastapi import FastAPI, Request
from fastapi.responses import Response

from services.traductor import TranslatePayload, translate

app = FastAPI()


@app.get('/')
async def health_check():
    return Response(content='OK', status_code=200)


@app.post('/translate')
async def handler_predict(payload: TranslatePayload):
    result = translate(payload)
    return result
