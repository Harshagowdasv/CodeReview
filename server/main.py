"""
server/main.py — FastAPI server for Code Review RL Environment
Fixed: /reset endpoint now accepts POST with no body (task_id defaults to "easy")
"""

from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
import random

app = FastAPI(title="Code Review RL Environment")

# ── Pydantic models ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"   # ← KEY FIX: optional with default

class StepRequest(BaseModel):
    action: str

# ── Diff bank (easy / medium / hard) ────────────────────────────────────────

TASKS = {
    "easy": {
        "diff": """\
--- a/utils.py
+++ b/utils.py
@@ -1,10 +1,10 @@
 def get_user(users, idx):
-    for i in range(len(users) + 1):   # off-by-one: raises IndexError
+    for i in range(len(users)):
         if users[i].id == idx:
             return users[i]
-    return users[0]                    # wrong fallback, should be None
+    return None

 def hash_password(pw):
-    return pw                          # plaintext – never store raw passwords
+    import hashlib
+    return hashlib.sha256(pw.encode()).hexdigest()
""",
        "ground_truth": [
            "off-by-one error in range",
            "wrong fallback return value",
            "plaintext password storage",
        ],
        "max_steps": 5,
    },
    "medium": {
        "diff": """\
--- a/api/orders.py
+++ b/api/orders.py
@@ -1,20 +1,20 @@
 import sqlite3

 def get_order(order_id):
     conn = sqlite3.connect("orders.db")
     cur = conn.cursor()
-    cur.execute(f"SELECT * FROM orders WHERE id = {order_id}")  # SQL injection
+    cur.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
     return cur.fetchone()

 def update_status(order_id, user_id, new_status):
-    if new_status in ["pending", "shipped", "delivered"]:  # missing auth check
+    order = get_order(order_id)
+    if order is None:
+        raise ValueError("Order not found")
+    if order["user_id"] != user_id:
+        raise PermissionError("Not your order")
+    if new_status in ["pending", "shipped", "delivered"]:
         _write_status(order_id, new_status)
-    # no else: silent failure on invalid status
+    else:
+        raise ValueError(f"Invalid status: {new_status}")
""",
        "ground_truth": [
            "sql injection",
            "missing authorization check",
            "silent failure on invalid status",
        ],
        "max_steps": 8,
    },
    "hard": {
        "diff": """\
--- a/auth/session.py
+++ b/auth/session.py
@@ -1,30 +1,30 @@
 import os, time, hashlib

 SESSION_STORE = {}

 def create_session(user_id):
-    token = hashlib.md5(str(user_id).encode()).hexdigest()  # weak token
+    token = os.urandom(32).hex()
     SESSION_STORE[token] = {"user_id": user_id, "created": time.time()}
     return token

 def validate_session(token):
     session = SESSION_STORE.get(token)
     if session is None:
         return None
-    # no expiry check — sessions last forever
+    if time.time() - session["created"] > 3600:
+        del SESSION_STORE[token]
+        return None
     return session["user_id"]

 def render_profile(user_input):
-    return f"<h1>Hello {user_input}</h1>"   # XSS vulnerability
+    import html
+    return f"<h1>Hello {html.escape(user_input)}</h1>"

 def load_file(path):
-    with open(path) as f:                   # path traversal
+    base = "/var/app/data/"
+    safe = os.path.realpath(os.path.join(base, path))
+    if not safe.startswith(base):
+        raise PermissionError("Path traversal blocked")
+    with open(safe) as f:
         return f.read()
""",
        "ground_truth": [
            "weak session token (md5)",
            "no session expiry",
            "xss vulnerability",
            "path traversal",
        ],
        "max_steps": 12,
    },
}

# ── In-memory state ──────────────────────────────────────────────────────────

state = {
    "step": 0,
    "current_task": None,
    "diff": "",
    "ground_truth_issues": [],
    "found_issues": [],
    "score": 0.0,
    "done": False,
}

# ── Helper ───────────────────────────────────────────────────────────────────

def compute_reward(action: str, ground_truth: list) -> tuple[float, list]:
    action_lower = action.lower()
    found = []
    for issue in ground_truth:
        keywords = issue.lower().split()
        if any(kw in action_lower for kw in keywords):
            found.append(issue)

    coverage = len(found) / max(len(ground_truth), 1)

    # Structure bonus: must contain SEVERITY, LINE, ISSUE, SUGGESTION
    required = ["severity:", "line:", "issue:", "suggestion:"]
    structure_bonus = 0.10 if all(r in action_lower for r in required) else 0.0

    # Length penalty
    length_penalty = -0.05 if len(action.strip()) < 30 else 0.0

    reward = min(1.0, coverage + structure_bonus + length_penalty)
    return round(reward, 4), found

# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(request: ResetRequest = Body(default=ResetRequest())):
    """
    Start a new episode. task_id is optional (defaults to 'easy').
    Accepts POST with no body, or with {"task_id": "easy|medium|hard"}.
    """
    task_id = (request.task_id or "easy").lower()
    if task_id not in TASKS:
        task_id = "easy"

    task = TASKS[task_id]

    state.update({
        "step": 0,
        "current_task": task_id,
        "diff": task["diff"],
        "ground_truth_issues": task["ground_truth"],
        "found_issues": [],
        "score": 0.0,
        "done": False,
    })

    observation = (
        f"[CODE REVIEW TASK - Difficulty: {task_id.upper()}]\n\n"
        f"Review the following diff and identify all issues:\n\n"
        f"DIFF:\n{task['diff']}"
    )

    return {
        "observation": observation,
        "state": {k: v for k, v in state.items() if k != "diff"},
    }


@app.post("/step")
def step(request: StepRequest):
    if state["done"]:
        return {
            "observation": "Episode is done. Call /reset to start a new episode.",
            "reward": 0.0,
            "done": True,
            "info": {},
        }

    state["step"] += 1
    task = TASKS[state["current_task"]]

    reward, newly_found = compute_reward(request.action, state["ground_truth_issues"])

    for issue in newly_found:
        if issue not in state["found_issues"]:
            state["found_issues"].append(issue)

    coverage = len(state["found_issues"]) / max(len(state["ground_truth_issues"]), 1)
    state["score"] = round(coverage, 4)

    done = (
        state["step"] >= task["max_steps"]
        or len(state["found_issues"]) == len(state["ground_truth_issues"])
    )
    state["done"] = done

    observation = (
        f"Step {state['step']} complete.\n"
        f"Issues found so far: {state['found_issues']}\n"
        f"Remaining issues: {[i for i in state['ground_truth_issues'] if i not in state['found_issues']]}\n\n"
        f"DIFF:\n{state['diff']}"
    )

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": {
            "issues_found": len(state["found_issues"]),
            "total_issues": len(state["ground_truth_issues"]),
            "coverage": state["score"],
            "step": state["step"],
        },
    }


@app.get("/state")
def get_state():
    return state


@app.get("/health")
def health():
    return {"status": "ok"}