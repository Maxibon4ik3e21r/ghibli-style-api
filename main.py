from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional
import requests
import time
import os

app = FastAPI()

# Переменные окружения
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
YOUR_API_KEY = os.getenv("YOUR_API_KEY")
REPLICATE_MODEL_VERSION = "166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f"

class ImageInput(BaseModel):
    image_url: HttpUrl
    prompt: Optional[str] = "GHIBLI anime style photo"

@app.post("/stylize")
def stylize(data: ImageInput, x_api_key: str = Header(None)):
    if x_api_key != YOUR_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "version": REPLICATE_MODEL_VERSION,
            "input": {
                "image": str(data.image_url),
                "prompt": data.prompt
            }
        }
    )

    if response.status_code != 201:
        raise HTTPException(status_code=response.status_code, detail="Replicate API error")

    prediction = response.json()
    prediction_id = prediction.get("id")
    status_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"

    # Проверка статуса
    for _ in range(30):
        result = requests.get(status_url, headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"})
        result_data = result.json()
        if result_data["status"] == "succeeded":
            return {"output": result_data["output"]}
        elif result_data["status"] == "failed":
            raise HTTPException(status_code=500, detail="Replicate processing failed")
        time.sleep(2)

    raise HTTPException(status_code=408, detail="Timeout: Replicate took too long")
