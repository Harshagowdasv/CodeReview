"""
FastAPI HTTP server wrapping the CodeReviewEnv.
Implements the OpenEnv standard API: /reset, /step, /state
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from env import CodeReviewEnv, Action, Observation, Reward, State

app = FastAPI(
    title="Code Review Assistant - OpenEnv",
    description="RL environment where an AI agent reviews code diffs and receives graded rewards.",
    version="1.0.0",
)

# Single shared environment instance (stateful per session)
env = CodeReviewEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


class StepRequest(BaseModel):
    action: str


@app.get("/")
def root():
    return {"name": "code-review-env", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest):
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    s = env.state()
    return {
        "step": s.step,
        "current_task": s.current_task,
        "diff": s.diff,
        "ground_truth_issues": s.ground_truth_issues,
        "found_issues": s.found_issues,
        "score": s.score,
        "done": s.done,
    }