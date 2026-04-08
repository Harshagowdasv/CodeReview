"""
Code Review Assistant - OpenEnv RL Environment
Real-world task: AI agent reviews code diffs and receives graded rewards
"""

import random
import re
from typing import Optional
from pydantic import BaseModel


# ──────────────────────────────────────────────
# Typed Models (OpenEnv spec)
# ──────────────────────────────────────────────

class Action(BaseModel):
    action: str


class Observation(BaseModel):
    observation: str


class Reward(BaseModel):
    reward: float
    done: bool
    info: dict


class State(BaseModel):
    step: int
    current_task: str
    diff: str
    ground_truth_issues: list
    found_issues: list
    score: float
    done: bool


# ──────────────────────────────────────────────
# Code Diff Dataset
# ──────────────────────────────────────────────

DIFFS = {
    "easy": [
        {
            "meta": "File: utils.py | Language: Python | PR: Fix user lookup",
            "diff": """\
--- a/utils.py
+++ b/utils.py
@@ -1,12 +1,12 @@
 def get_user_by_id(users, user_id):
-    for i in range(len(users)):
-        if users[i]["id"] == user_id:
-            return users[i]
+    for i in range(len(users) + 1):
+        if users[i]["id"] == user_id:
+            return users[i]
     return None

 def calculate_discount(price, discount):
-    return price - (price * discount)
+    return price - (price * discount / 100)
""",
            "issues": [
                {"severity": "critical", "line": "4", "keyword": "off-by-one", "alt": ["index out of range", "range len+1", "out of bounds"]},
                {"severity": "info",     "line": "9", "keyword": "discount",   "alt": ["division", "percentage", "100"]},
            ],
        },
        {
            "meta": "File: auth.py | Language: Python | PR: Password validation",
            "diff": """\
--- a/auth.py
+++ b/auth.py
@@ -1,10 +1,10 @@
 def validate_password(password):
-    if len(password) > 8:
+    if len(password) >= 8:
         return True
     return False

 def login(username, password):
     user = db.find_user(username)
-    if user and user.password == password:
+    if user.password == password:
         return generate_token(user)
     return None
""",
            "issues": [
                {"severity": "major", "line": "7", "keyword": "null",  "alt": ["none check", "user is none", "attributeerror", "no null check"]},
                {"severity": "major", "line": "6", "keyword": "plain", "alt": ["hash", "bcrypt", "plaintext", "password comparison", "not hashed"]},
            ],
        },
    ],
    "medium": [
        {
            "meta": "File: api/orders.py | Language: Python | PR: Add order processing",
            "diff": """\
--- a/api/orders.py
+++ b/api/orders.py
@@ -1,35 +1,35 @@
 def process_order(order_id, user_id):
     order = db.get_order(order_id)
-    if order.user_id != user_id:
-        raise PermissionError("Not your order")
+    # removed auth check for speed
     items = order.items
     total = 0
     for item in items:
-        total += item.price * item.quantity
+        total += item.price * item.quantity * 1.0
     if total > 1000:
         discount = total * 0.1
-        total = total - discount
+        total -= discount
         log_discount(user_id, discount)
     charge_user(user_id, total)
     order.status = "processed"
-    db.save(order)
+    db.save(order)
+    db.save(order)
     return order

 def get_order_history(user_id, page=1):
-    orders = db.query(f"SELECT * FROM orders WHERE user_id={user_id}")
+    orders = db.query(f"SELECT * FROM orders WHERE user_id={user_id} LIMIT 100")
     return orders
""",
            "issues": [
                {"severity": "critical", "line": "3",  "keyword": "authorization",      "alt": ["auth check", "permission", "removed auth", "idor", "access control"]},
                {"severity": "critical", "line": "19", "keyword": "sql injection",       "alt": ["f-string query", "unsanitized", "sql", "injection"]},
                {"severity": "major",    "line": "15", "keyword": "duplicate save",      "alt": ["db.save called twice", "double save", "redundant"]},
                {"severity": "minor",    "line": "8",  "keyword": "multiply 1.0",        "alt": ["unnecessary", "float cast", "redundant multiplication"]},
            ],
        },
    ],
    "hard": [
        {
            "meta": "File: server/upload.py, server/session.py | Language: Python | PR: File upload + session handling",
            "diff": """\
--- a/server/upload.py
+++ b/server/upload.py
@@ -1,40 +1,40 @@
 import os
 import subprocess

 def handle_upload(file, user_id):
     filename = file.filename
-    safe_name = secure_filename(filename)
-    path = os.path.join(UPLOAD_DIR, safe_name)
+    path = os.path.join(UPLOAD_DIR, filename)
     file.save(path)
-    if filename.endswith(('.jpg','.png','.gif')):
+    if filename.endswith(('.jpg','.png','.gif','.svg')):
         subprocess.run(['convert', path, path])
     return {"url": f"/uploads/{filename}"}

 def get_file_url(filename):
-    return f"/uploads/{secure_filename(filename)}"
+    return f"/uploads/{filename}"

--- a/server/session.py
+++ b/server/session.py
@@ -1,25 +1,25 @@
 import time
 SESSION_STORE = {}

 def create_session(user_id):
-    token = secrets.token_hex(32)
+    token = str(user_id) + str(time.time())
     SESSION_STORE[token] = {"user_id": user_id, "created": time.time()}
     return token

 def validate_session(token):
     session = SESSION_STORE.get(token)
-    if session and time.time() - session["created"] < 3600:
+    if session:
         return session["user_id"]
     return None
""",
            "issues": [
                {"severity": "critical", "line": "7",  "keyword": "path traversal",    "alt": ["directory traversal", "secure_filename removed", "unsafe filename", "../"]},
                {"severity": "critical", "line": "9",  "keyword": "svg",               "alt": ["xss", "svg upload", "script injection", "svg xss"]},
                {"severity": "critical", "line": "21", "keyword": "predictable token", "alt": ["weak token", "not random", "secrets", "user_id token", "time token"]},
                {"severity": "critical", "line": "27", "keyword": "session expiry",    "alt": ["no expiry", "session never expires", "timeout removed"]},
                {"severity": "major",    "line": "10", "keyword": "subprocess",        "alt": ["command injection", "shell", "arbitrary command"]},
                {"severity": "major",    "line": "14", "keyword": "path traversal url","alt": ["secure_filename", "url filename unsafe"]},
            ],
        },
    ],
}


# ──────────────────────────────────────────────
# Reward Logic
# ──────────────────────────────────────────────

def score_review(action_text: str, ground_truth_issues: list) -> tuple[float, list]:
    text_lower = action_text.lower()
    found = []

    for issue in ground_truth_issues:
        matched = False
        if issue["keyword"] in text_lower:
            matched = True
        else:
            for alt in issue.get("alt", []):
                if alt in text_lower:
                    matched = True
                    break
        if matched:
            found.append(issue["keyword"])

    if not ground_truth_issues:
        return 0.0, found

    base_score = len(found) / len(ground_truth_issues)

    structure_bonus = 0.0
    if re.search(r"severity\s*:", text_lower) and re.search(r"issue\s*:", text_lower):
        structure_bonus = 0.1

    length_penalty = 0.0
    if len(found) > 0 and len(action_text) < 50 * len(found):
        length_penalty = 0.05

    final = min(1.0, base_score + structure_bonus - length_penalty)
    return round(final, 4), found


# ──────────────────────────────────────────────
# Environment Class
# ──────────────────────────────────────────────

class CodeReviewEnv:
    def __init__(self):
        self._state: Optional[State] = None
        self._task_id: str = "easy"
        self._max_steps = {"easy": 5, "medium": 8, "hard": 12}

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in DIFFS:
            task_id = "easy"

        diff_pool = DIFFS[task_id]
        chosen = random.choice(diff_pool)

        self._task_id = task_id
        self._state = State(
            step=0,
            current_task=task_id,
            diff=chosen["diff"],
            ground_truth_issues=chosen["issues"],
            found_issues=[],
            score=0.0,
            done=False,
        )

        obs_text = (
            f"[CODE REVIEW TASK - Difficulty: {task_id.upper()}]\n"
            f"{chosen['meta']}\n\n"
            f"Please review the following diff and identify all bugs, security issues, "
            f"and code quality problems. For each issue, respond in this format:\n"
            f"SEVERITY: <critical|major|minor|info> | LINE: <line> | ISSUE: <description> | SUGGESTION: <fix>\n\n"
            f"DIFF:\n{chosen['diff']}"
        )
        return Observation(observation=obs_text)

    def step(self, action: Action) -> tuple[Observation, Reward]:
        if self._state is None or self._state.done:
            raise ValueError("Call reset() first.")

        self._state.step += 1
        score, found = score_review(action.action, self._state.ground_truth_issues)
        self._state.found_issues = found
        self._state.score = score

        max_steps = self._max_steps.get(self._task_id, 5)
        done = score >= 0.9 or self._state.step >= max_steps
        self._state.done = done

        reward = Reward(
            reward=score,
            done=done,
            info={
                "step": self._state.step,
                "issues_found": len(found),
                "total_issues": len(self._state.ground_truth_issues),
                "found_keywords": found,
                "task": self._task_id,
            }
        )

        next_obs = Observation(
            observation=(
                f"[STEP {self._state.step}] Review received. "
                f"Issues identified so far: {len(found)}/{len(self._state.ground_truth_issues)}. "
                f"Score: {score:.2f}. "
                + ("Task complete!" if done else "Continue reviewing or refine your analysis.")
            )
        )
        return next_obs, reward

    def state(self) -> State:
        if self._state is None:
            return State(
                step=0, current_task="", diff="",
                ground_truth_issues=[], found_issues=[],
                score=0.0, done=False
            )
        return self._state