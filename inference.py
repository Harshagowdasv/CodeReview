"""
Inference Script - MANDATORY for ScalerHack OpenEnv submission.

Uses the OpenAI-compatible client with Hugging Face Router API.
Environment variables required:
  - API_BASE_URL : HF router endpoint (default: https://router.huggingface.co/v1)
  - MODEL_NAME   : Model identifier
  - HF_TOKEN     : Your Hugging Face API key
  - ENV_URL      : Running environment URL
"""

import os
import sys
import re

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not found. Install with: pip install openai")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[ERROR] requests package not found. Install with: pip install requests")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN", os.getenv("API_KEY", ""))
ENV_URL          = os.getenv("ENV_URL",      "http://localhost:7860")

MAX_STEPS    = 8
TEMPERATURE  = 0.2
MAX_TOKENS   = 1024
FALLBACK_ACTION  = "SEVERITY: info | LINE: 1 | ISSUE: Unable to parse diff | SUGGESTION: noop()"
TASKS        = ["easy", "medium", "hard"]

# ── Client ───────────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be shown a code diff (unified diff format).
Your job is to identify ALL bugs, security vulnerabilities, logic errors, and code quality issues.

For EACH issue found, respond with exactly this format (one per line):
SEVERITY: <critical|major|minor|info> | LINE: <line_number> | ISSUE: <clear description> | SUGGESTION: <concrete fix>

Rules:
- Be specific about the line number from the diff
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
        print(f"[WARN] LLM call failed: {exc}. Using fallback action.")
        return FALLBACK_ACTION


def run_task(env_url: str, task_id: str) -> float:
    """Run one full episode. Returns final reward."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    try:
        reset_resp = requests.post(
            f"{env_url}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()["observation"]
        print(f"[RESET] Got diff ({len(observation)} chars)")
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}")
        return 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        messages.append({"role": "user", "content": observation})
        action = call_llm(messages)
        print(f"\n[STEP {step}] Action preview: {action[:200]}")
        messages.append({"role": "assistant", "content": action})

        try:
            step_resp = requests.post(
                f"{env_url}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            result     = step_resp.json()
            observation = result["observation"]
            reward      = result["reward"]
            done        = result["done"]
            info        = result.get("info", {})
            final_reward = reward
            print(f"[STEP {step}] Reward: {reward:.2f} | Found: {info.get('issues_found',0)}/{info.get('total_issues',0)} | Done: {done}")
            if done:
                break
        except Exception as e:
            print(f"[ERROR] Step failed: {e}")
            break

    print(f"\n[RESULT] Task '{task_id}' final reward: {final_reward:.2f}")
    return final_reward


def main():
    print(f"Code Review Assistant — Inference Script")
    print(f"Model  : {MODEL_NAME}")
    print(f"Server : {ENV_URL}")

    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(ENV_URL, task_id)
        except Exception as e:
            print(f"[ERROR] Task '{task_id}' failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    for task, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:8s}: [{bar}] {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':8s}: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()