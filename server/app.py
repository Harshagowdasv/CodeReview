from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
from env import CodeReviewEnv, Action

app = FastAPI()

env = CodeReviewEnv()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: str

@app.get("/")
def home():
    return {"message": "CodeReview OpenEnv running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(request: ResetRequest = Body(default=ResetRequest())):
    obs = env.reset(task_id=request.task_id or "easy")
    return {"observation": obs.observation}

@app.post("/step")
def step(request: StepRequest):
    try:
        action = Action(action=request.action)
        obs, reward = env.step(action)
        return {
            "observation": obs.observation,
            "reward": reward.reward,
            "done": reward.done,
            "info": reward.info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    s = env.state()
    return {
        "step": s.step,
        "score": s.score,
        "done": s.done
    }