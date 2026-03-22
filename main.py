from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

from agent.react_agent import ReactAgent

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建全局Agent实例
agent = ReactAgent()

# 请求模型
class QueryRequest(BaseModel):
    query: str
    thread_id: str = None

# 流式响应
@app.post("/api/query")
async def query_agent(request: QueryRequest):
    print(f"Received request: {request}")
    async def generate():
        try:
            for chunk in agent.excute_stream(request.query, request.thread_id):
                yield chunk
                await asyncio.sleep(0.1)  # 避免过快发送
        except Exception as e:
            print(f"Error: {str(e)}")
            yield f"Error: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
