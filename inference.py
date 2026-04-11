"""
Inference Script - MANDATORY for ScalerHack OpenEnv submission.
Prints [START]/[STEP]/[END] structured blocks required by the validator.
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not found. Install with: pip install openai", flush=True)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[ERROR] requests package not found. Install with: pip install requests", flush=True)
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", os.getenv("API_KEY", ""))
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS       = 8
TEMPERATURE     = 0.2
MAX_TOKENS      = 1024
FALLBACK_ACTION = "SEVERITY: info | LINE: 1 | ISSUE: Unable to parse diff | SUGGESTION: noop()"
TASKS           = ["easy", "medium", "hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be shown a code diff (unified diff format).
Your job is to identify ALL bugs, security vulnerabilities, logic errors, and code quality issues.

For EACH issue found, respond with exactly this format (one per line):
SEVERITY: <critical|major|minor|info> | LINE: <line_number> | ISSUE: <clear description> | SUGGESTION: <concrete fix>

Rules:
- critical = security vulnerabilities or crashes
- major = logic bugs or missing checks
- minor = code quality, style, performance
- info = nitpicks or improvements
- Cover ALL issues you can find
- Do not add any preamble or explanation outside the format
"""


def call_llm(messages: list) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or FALLBACK_ACTION
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Using fallback.", flush=True)
        return FALLBACK_ACTION


def run_task(env_url: str, task_id: str) -> dict:
    """Run one full episode. Returns dict with score and steps."""

    # ── [START] block ──────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    try:
        reset_resp = requests.post(
            f"{env_url}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()["observation"]
    except Exception as e:
        print(f"[WARN] Reset failed: {e}", flush=True)
        print(f"[STEP] step=1 reward=0.01", flush=True)
        print(f"[END] task={task_id} score=0.01 steps=1", flush=True)
        return {"score": 0.01, "steps": 1}

    messages     = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_reward = 0.01
    step_count   = 0

    for step in range(1, MAX_STEPS + 1):
        step_count = step
        messages.append({"role": "user", "content": observation})
        action = call_llm(messages)
        messages.append({"role": "assistant", "content": action})

        try:
            step_resp = requests.post(
                f"{env_url}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            result       = step_resp.json()
            observation  = result["observation"]
            reward       = float(result["reward"])
            done         = result["done"]
            final_reward = reward

            # ── [STEP] block ───────────────────────────────────
            print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

            if done:
                break
        except Exception as e:
            print(f"[WARN] Step {step} failed: {e}", flush=True)
            print(f"[STEP] step={step} reward=0.01", flush=True)
            break

    # ── [END] block ────────────────────────────────────────────
    print(f"[END] task={task_id} score={final_reward:.4f} steps={step_count}", flush=True)
    return {"score": final_reward, "steps": step_count}


def main():
    print(f"Code Review Assistant — Inference Script", flush=True)
    print(f"Model  : {MODEL_NAME}", flush=True)
    print(f"Server : {ENV_URL}", flush=True)

    results = {}
    for task_id in TASKS:
        try:
            results[task_id] = run_task(ENV_URL, task_id)
        except Exception as e:
            print(f"[WARN] Task '{task_id}' crashed: {e}", flush=True)
            print(f"[START] task={task_id}", flush=True)
            print(f"[STEP] step=1 reward=0.01", flush=True)
            print(f"[END] task={task_id} score=0.01 steps=1", flush=True)
            results[task_id] = {"score": 0.01, "steps": 1}

    scores = [v["score"] for v in results.values()]
    avg    = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAll tasks complete. Average score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()