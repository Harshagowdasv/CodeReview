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


# ✅ UI (SAFE — won't break OpenEnv)
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>CodeReview OpenEnv</title>
        <style>
            body { font-family: Arial; background:#0d1117; color:#fff; padding:20px;}
            button { padding:10px; margin:5px; }
            textarea { width:100%; height:120px; }
            pre { background:#111; padding:10px; }
        </style>
    </head>
    <body>
        <h1>🚀 Code Review OpenEnv</h1>

        <button onclick="reset()">Reset</button>
        <button onclick="state()">State</button>

        <br><br>
        <textarea id="action" placeholder="Write review..."></textarea>
        <br>
        <button onclick="step()">Submit</button>

        <h3>Output:</h3>
        <pre id="output"></pre>

        <script>
            async function reset() {
                let res = await fetch('/reset', {method:'POST'});
                let data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }

            async function step() {
                let action = document.getElementById('action').value;
                let res = await fetch('/step', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({action})
                });
                let data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }

            async function state() {
                let res = await fetch('/state');
                let data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
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