import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from sglang.utils import execute_shell_command, wait_for_server, print_highlight


app = FastAPI()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def launch_model_server():
    try:

        execute_shell_command(
            "python -m sglang.launch_server --model-path ./Meta-Llama-3.1-8B-Instruct --port 8004 --host 10.201.8.114"
        )

        wait_for_server("http://10.201.8.114:8005")
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

launch_model_server()

client = openai.Client(base_url="http://10.201.8.114:8004/v1", api_key="None")

# 推理接口
@app.post("/v1/completion", response_model=CompletionResponse)
async def generate_response(request: CompletionRequest):
    try:

        response = client.chat.completions.create(
            model="./Meta-Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return CompletionResponse(response=response["choices"][0]["message"]["content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.201.8.114", port=8004)

