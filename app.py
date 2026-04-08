from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from env import CodeReviewEnv, Action

app = FastAPI()

env = CodeReviewEnv()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: str


# UI (safe)
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="background:#0d1117;color:white;font-family:Arial;padding:20px;">
        <h1>🚀 CodeReview OpenEnv</h1>
        <button onclick="reset()">Reset</button>
        <button onclick="state()">State</button><br><br>
        <textarea id="action" style="width:100%;height:120px;"></textarea><br>
        <button onclick="step()">Submit</button>
        <pre id="output"></pre>

        <script>
            async function reset(){
                let r=await fetch('/reset',{method:'POST'});
                let d=await r.json();
                output.innerText=JSON.stringify(d,null,2);
            }
            async function step(){
                let action=document.getElementById('action').value;
                let r=await fetch('/step',{
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({action})
                });
                let d=await r.json();
                output.innerText=JSON.stringify(d,null,2);
            }
            async function state(){
                let r=await fetch('/state');
                let d=await r.json();
                output.innerText=JSON.stringify(d,null,2);
            }
        </script>
    </body>
    </html>
    """


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


# ✅ REQUIRED FOR OPENENV
def main():
    return app