# 🔍 Code Review Assistant — OpenEnv RL Environment

A real-world reinforcement learning environment where an AI agent reviews **code diffs** (pull requests), identifies bugs and security issues, and receives **graded rewards** based on review quality.

Built for the **ScalerHack OpenEnv Hackathon** (Meta × PyTorch × Hugging Face).

---

## 🎯 Task Description

The agent is shown a **unified code diff** and must identify all problems in a structured format:

```
SEVERITY: <critical|major|minor|info> | LINE: <line> | ISSUE: <description> | SUGGESTION: <fix>
```

Examples of real-world issues the agent must detect:
- Off-by-one errors, null pointer dereferences
- SQL injection, path traversal, XSS vulnerabilities  
- Authorization bypass, weak session tokens
- Duplicate operations, redundant code

---

## 📊 Difficulty Levels & Tasks

| Task | Lines | Issues | Max Steps | Description |
|------|-------|--------|-----------|-------------|
| `easy`   | 5–20  | 1–2 | 5  | Simple bugs in short Python functions |
| `medium` | 20–60 | 2–4 | 8  | Logic errors, SQL injection, auth issues |
| `hard`   | 60–150| 4–6 | 12 | Security vulnerabilities, architecture flaws |

---

## 🔧 Action / Observation / State

### Action (string)
```
SEVERITY: critical | LINE: 7 | ISSUE: Missing null check before user.password access | SUGGESTION: Add 'if user is None: return None' before line 7
SEVERITY: major | LINE: 6 | ISSUE: Plaintext password comparison | SUGGESTION: Use bcrypt.checkpw(password, user.hashed_password)
```

### Observation (string)
```
[CODE REVIEW TASK - Difficulty: MEDIUM]
File: api/orders.py | Language: Python | PR: Add order processing

Please review the following diff...

DIFF:
--- a/api/orders.py
+++ b/api/orders.py
...
```

### State (object)
```json
{
  "step": 2,
  "current_task": "medium",
  "diff": "...",
  "ground_truth_issues": [...],
  "found_issues": ["sql injection", "authorization"],
  "score": 0.75,
  "done": false
}
```

---

## 🏆 Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Issue coverage | 0.0–1.0 | Fraction of ground-truth issues correctly identified |
| Structure bonus | +0.10 | Follows `SEVERITY: ... \| LINE: ... \| ISSUE: ...` format |
| Length penalty | −0.05 | Review too short relative to issues found |

Reward range: **0.0 – 1.0** (continuous, meaningful progress signal at every step)

---

## 🚀 Setup & Running

### Local (Docker)

```bash
# Build
docker build -t code-review-env .

# Run
docker run -p 7860:7860 code-review-env
```

### Local (Python)

```bash
pip install -r requirements.txt
cd server
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Test the API

```bash
# Reset (start a new episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Step (send a review)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "SEVERITY: critical | LINE: 4 | ISSUE: Off-by-one error, range(len+1) causes IndexError | SUGGESTION: Change to range(len(users))"}'

# State
curl http://localhost:7860/state
```

---

## 🤖 Inference Script

```bash
# Set required env vars
export HF_TOKEN="your_hf_token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
export ENV_URL="http://localhost:7860"

# Run
python inference.py
```

---

## 📁 Project Structure

```
code-review-env/
├── Dockerfile          # Root Dockerfile (required)
├── openenv.yaml        # OpenEnv spec
├── requirements.txt
├── inference.py        # Mandatory inference script
└── server/
    ├── main.py         # FastAPI HTTP server
    └── env.py          # Core RL environment logic
```

---

## 🔗 Links

- **Hugging Face Space**: _[add your HF Space link here]_
- **OpenEnv Docs**: https://github.com/huggingface/open-env