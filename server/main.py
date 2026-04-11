"""
FastAPI HTTP server wrapping the CodeReviewEnv.
Implements the OpenEnv standard API: /reset, /step, /state
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
try:
    from server.env import CodeReviewEnv, Action, Observation, Reward, State
except ImportError:
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


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Code Review Assistant — OpenEnv</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }
    header { background: linear-gradient(135deg, #1f6feb, #388bfd); padding: 32px 24px; text-align: center; }
    header h1 { font-size: 2rem; font-weight: 700; }
    header p  { margin-top: 8px; opacity: .85; font-size: 1rem; }
    .badge { display:inline-block; background:#238636; color:#fff; border-radius:20px; padding:4px 14px; font-size:.8rem; margin-top:12px; }
    main { max-width: 900px; margin: 32px auto; padding: 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap: 16px; margin-bottom: 28px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }
    .card h3 { font-size: 1rem; color: #58a6ff; margin-bottom: 8px; }
    .card p  { font-size: .875rem; color: #8b949e; line-height: 1.5; }
    .card .tag { display:inline-block; font-size:.75rem; padding:2px 10px; border-radius:12px; margin-top:10px; font-weight:600; }
    .easy   { background:#1a3a1a; color:#3fb950; }
    .medium { background:#3a2a0a; color:#d29922; }
    .hard   { background:#3a0a0a; color:#f85149; }

    section h2 { font-size: 1.2rem; margin-bottom: 14px; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    .api-table { width:100%; border-collapse:collapse; font-size:.875rem; }
    .api-table th { background:#21262d; padding:10px 14px; text-align:left; color:#8b949e; border-bottom:1px solid #30363d; }
    .api-table td { padding:10px 14px; border-bottom:1px solid #21262d; vertical-align:top; }
    .api-table td:first-child { color:#58a6ff; font-family:monospace; font-size:.95rem; white-space:nowrap; }
    .method { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.75rem; font-weight:700; margin-right:6px; }
    .get  { background:#0d419d; color:#58a6ff; }
    .post { background:#1a3a1a; color:#3fb950; }

    .try-box { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px; margin-top:28px; }
    .try-box h2 { font-size:1.1rem; color:#58a6ff; margin-bottom:16px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; align-items:center; }
    select, button, textarea { font-size:.875rem; border-radius:8px; border:1px solid #30363d; background:#0d1117; color:#e6edf3; padding:8px 12px; }
    button { background:#1f6feb; border-color:#1f6feb; cursor:pointer; font-weight:600; padding:8px 18px; }
    button:hover { background:#388bfd; }
    textarea { width:100%; height:120px; resize:vertical; font-family:monospace; font-size:.8rem; }
    pre { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:14px; font-size:.78rem; overflow-x:auto; white-space:pre-wrap; word-break:break-word; color:#e6edf3; max-height:320px; overflow-y:auto; }
    .reward-bar-wrap { margin-top:10px; }
    .reward-label { font-size:.85rem; color:#8b949e; margin-bottom:4px; }
    .reward-bar-bg { background:#21262d; border-radius:6px; height:14px; width:100%; }
    .reward-bar-fill { height:14px; border-radius:6px; background: linear-gradient(90deg,#f85149,#d29922,#3fb950); transition:width .5s; }
    footer { text-align:center; padding:24px; color:#8b949e; font-size:.8rem; border-top:1px solid #30363d; margin-top:40px; }
  </style>
</head>
<body>
<header>
  <h1>🔍 Code Review Assistant</h1>
  <p>Real-world RL environment — AI agent reviews code diffs &amp; earns rewards for finding bugs</p>
  <span class="badge">✅ openenv · ScalerHack 2025</span>
</header>

<main>
  <div class="grid" style="margin-top:24px;">
    <div class="card">
      <h3>🟢 Easy</h3>
      <p>Short Python diffs (5–20 lines) with 1–2 obvious bugs: off-by-one errors, null pointer issues.</p>
      <span class="tag easy">Max 5 steps</span>
    </div>
    <div class="card">
      <h3>🟡 Medium</h3>
      <p>Multi-function diffs (20–60 lines) with logic errors, SQL injection, missing auth checks.</p>
      <span class="tag medium">Max 8 steps</span>
    </div>
    <div class="card">
      <h3>🔴 Hard</h3>
      <p>Complex diffs (60–150 lines) with security vulnerabilities, race conditions, architecture flaws.</p>
      <span class="tag hard">Max 12 steps</span>
    </div>
  </div>

  <section>
    <h2>📡 API Endpoints</h2>
    <table class="api-table">
      <thead><tr><th>Endpoint</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><span class="method post">POST</span>/reset</td><td>Start a new episode. Body: <code>{"task_id": "easy|medium|hard"}</code></td></tr>
        <tr><td><span class="method post">POST</span>/step</td><td>Submit a review action. Body: <code>{"action": "SEVERITY: critical | LINE: 4 | ISSUE: ... | SUGGESTION: ..."}</code></td></tr>
        <tr><td><span class="method get">GET</span>/state</td><td>Get current environment state (step, score, issues found)</td></tr>
        <tr><td><span class="method get">GET</span>/health</td><td>Health check — returns <code>{"status": "ok"}</code></td></tr>
      </tbody>
    </table>
  </section>

  <div class="try-box">
    <h2>🧪 Try it Live</h2>
    <div class="row">
      <label>Task:</label>
      <select id="taskSel"><option>easy</option><option>medium</option><option>hard</option></select>
      <button onclick="doReset()">▶ Reset Episode</button>
      <button onclick="doState()">📊 Get State</button>
    </div>
    <div id="obsBox" style="display:none;margin-bottom:12px;">
      <div class="reward-label">Observation:</div>
      <pre id="obsOut"></pre>
      <div class="reward-label" style="margin-top:10px;">Your Review (one issue per line):</div>
      <textarea id="actionIn" placeholder="SEVERITY: critical | LINE: 4 | ISSUE: Off-by-one error in range | SUGGESTION: Change range(len+1) to range(len(users))"></textarea>
      <button onclick="doStep()" style="margin-top:8px;">📨 Submit Review</button>
    </div>
    <div id="rewardBox" style="display:none;" class="reward-bar-wrap">
      <div class="reward-label">Reward: <strong id="rewardVal">0.00</strong></div>
      <div class="reward-bar-bg"><div class="reward-bar-fill" id="rewardFill" style="width:0%"></div></div>
    </div>
    <pre id="out">← Click "Reset Episode" to start</pre>
  </div>
</main>

<footer>Built for ScalerHack · OpenEnv · Meta × PyTorch × Hugging Face</footer>

<script>
  const base = '';
  const out  = document.getElementById('out');
  const obs  = document.getElementById('obsOut');

  async function doReset() {
    const task = document.getElementById('taskSel').value;
    out.textContent = 'Resetting...';
    const r = await fetch(base+'/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task_id:task})});
    const d = await r.json();
    obs.textContent = d.observation;
    document.getElementById('obsBox').style.display = 'block';
    document.getElementById('rewardBox').style.display = 'none';
    out.textContent = JSON.stringify(d, null, 2);
  }

  async function doStep() {
    const action = document.getElementById('actionIn').value.trim();
    if (!action) { alert('Write a review first!'); return; }
    out.textContent = 'Submitting...';
    const r = await fetch(base+'/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action})});
    const d = await r.json();
    obs.textContent = d.observation;
    const pct = Math.round((d.reward||0)*100);
    document.getElementById('rewardBox').style.display = 'block';
    document.getElementById('rewardVal').textContent = (d.reward||0).toFixed(4);
    document.getElementById('rewardFill').style.width = pct+'%';
    out.textContent = JSON.stringify(d, null, 2);
  }

  async function doState() {
    const r = await fetch(base+'/state');
    const d = await r.json();
    out.textContent = JSON.stringify(d, null, 2);
  }
</script>
</body>
</html>
""")



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        task_id = body.get("task_id", "easy") if body else "easy"
    except Exception:
        task_id = "easy"
    obs = env.reset(task_id=task_id or "easy")
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