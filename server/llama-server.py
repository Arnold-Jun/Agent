import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import httpx  
from sglang.utils import execute_shell_command, wait_for_server, print_highlight

app = FastAPI()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_SERVER_URL = "http://10.201.8.114:8004/v1"
MODEL_NAME = "./Meta-Llama-3.1-8B-Instruct"

def launch_model_server():
    try:
        execute_shell_command(
            "python -m sglang.launch_server --model-path ./Meta-Llama-3.1-8B-Instruct --port 8004 --host 10.201.8.114"
        )
        wait_for_server("http://10.201.8.114:8004")
        print_highlight("Server launched successfully!")
    except Exception as e:
        print(f"Failed to launch the model server: {e}")
        sys.exit(1)

class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 64

class CompletionResponse(BaseModel):
    response: str

# 启动服务器（后台异步）
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, launch_model_server)

app.add_event_handler("startup", startup_event)

# 异步 HTTP 客户端
async def async_generate_response(prompt: str, temperature: float, max_tokens: int):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{MODEL_SERVER_URL}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=10.0  # 设置超时时间，避免长时间阻塞
            )
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in generating response: {str(e)}")

# 异步推理接口
@app.post("/v1/completion", response_model=CompletionResponse)
async def generate_response(request: CompletionRequest):
    response_text = await async_generate_response(request.prompt, request.temperature, request.max_tokens)
    return CompletionResponse(response=response_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.201.8.114", port=8004)
