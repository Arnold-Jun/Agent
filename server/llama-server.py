import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import httpx
from sglang.utils import execute_shell_command, wait_for_server, print_highlight
from typing import List

app = FastAPI()
MODEL_SERVER_URL = "http://10.201.8.114:8004/v1"
MODEL_NAME = "./Meta-Llama-3.1-8B-Instruct"

BATCH_SIZE = 5  # 每批处理5个请求
BATCH_INTERVAL = 0.2  # 200ms 内合并请求
request_queue = asyncio.Queue()  # 请求队列
responses = {}  # 存储返回的结果

class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 64

class CompletionResponse(BaseModel):
    response: str

# 启动 LLM 服务器
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, launch_model_server)

app.add_event_handler("startup", startup_event)

# 批量推理任务（独立后台运行）
async def batch_worker():
    while True:
        await asyncio.sleep(BATCH_INTERVAL)  # 等待时间，积累请求
        batch = []
        tasks = []

        # 从队列中取出最多 `BATCH_SIZE` 个请求
        for _ in range(BATCH_SIZE):
            if request_queue.empty():
                break
            request_id, request_data = await request_queue.get()
            batch.append(request_data)
            tasks.append(request_id)

        if not batch:
            continue  # 没有请求则跳过

        # 批量推理
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{MODEL_SERVER_URL}/chat/completions",
                    json={
                        "model": MODEL_NAME,
                        "messages": [{"role": "user", "content": data["prompt"]} for data in batch],
                        "temperature": batch[0]["temperature"], 
                        "max_tokens": batch[0]["max_tokens"],
                    },
                    timeout=15.0
                )
                response_json = response.json()
                results = [resp["message"]["content"] for resp in response_json["choices"]]

                # 结果拆分存储
                for task_id, result in zip(tasks, results):
                    responses[task_id] = result
            except Exception as e:
                for task_id in tasks:
                    responses[task_id] = f"Error: {str(e)}"

# 启动批量任务
@app.on_event("startup")
async def start_batch_worker():
    asyncio.create_task(batch_worker())

# 异步推理接口（进入队列）
@app.post("/v1/completion", response_model=CompletionResponse)
async def generate_response(request: CompletionRequest, background_tasks: BackgroundTasks):
    request_id = str(id(request))
    await request_queue.put((request_id, request.dict()))
    
    while request_id not in responses:
        await asyncio.sleep(0.05)  

    response_text = responses.pop(request_id)
    return CompletionResponse(response=response_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.201.8.114", port=8004)

