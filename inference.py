"""
Inference Script - MANDATORY for ScalerHack OpenEnv submission.

Uses the OpenAI-compatible client with Hugging Face Router API.
Environment variables required:
  - API_BASE_URL : HF router endpoint (default: https://router.huggingface.co/v1)
  - MODEL_NAME   : Model identifier (e.g. meta-llama/Llama-3.3-70B-Instruct)
  - HF_TOKEN     : Your Hugging Face API key
  - ENV_URL      : Environment server URL
"""

import os
import time
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS    = 8
TEMPERATURE  = 0.2
MAX_TOKENS   = 1024
RETRY_COUNT  = 2
DEBUG        = True

FALLBACK_ACTION = (
    "SEVERITY: info | LINE: 1 | ISSUE: Unable to parse diff | SUGGESTION: noop()"
)

TASKS = ["easy", "medium", "hard"]

# ── Validate Environment ─────────────────────────────────────────────────────
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN is missing! Add it in Hugging Face Secrets.")

print(f"✅ Using Model: {MODEL_NAME}")
print(f"🌐 API Base: {API_BASE_URL}")
print(f"🧪 ENV URL: {ENV_URL}")

# ── Client Setup ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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

# ── LLM Call with Retry ──────────────────────────────────────────────────────
def call_llm(messages: list) -> str:
    for attempt in range(RETRY_COUNT):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content or FALLBACK_ACTION

        except Exception as exc:
            print(f"⚠️ LLM call failed (attempt {attempt+1}): {exc}")
            time.sleep(1)

    print("❌ All retries failed. Using fallback.")
    return FALLBACK_ACTION


# ── Run Task ─────────────────────────────────────────────────────────────────
def run_task(env_url: str, task_id: str) -> float:
    print(f"\n{'='*60}")
    print(f"🚀 TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset
    reset_resp = requests.post(
        f"{env_url}/reset", json={"task_id": task_id}, timeout=30
    )
    reset_resp.raise_for_status()

    observation = reset_resp.json()["observation"]
    print(f"[RESET] Diff size: {len(observation)} chars")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        messages.append({"role": "user", "content": observation})

        action = call_llm(messages)

        if DEBUG:
            print(f"\n[STEP {step}] Action:\n{action[:300]}{'...' if len(action)>300 else ''}")

        messages.append({"role": "assistant", "content": action})

        # Step
        step_resp = requests.post(
            f"{env_url}/step",
            json={"action": action},
            timeout=30,
        )
        step_resp.raise_for_status()

        result = step_resp.json()

        observation = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        final_reward = reward

        print(
            f"[STEP {step}] Reward: {reward:.2f} | "
            f"Issues: {info.get('issues_found',0)}/{info.get('total_issues',0)} | Done: {done}"
        )

        if done:
            print(f"✅ Finished at step {step}")
            break

    print(f"\n🎯 FINAL SCORE ({task_id}): {final_reward:.2f}")
    return final_reward


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n🧠 Code Review Agent - Starting...\n")

    scores = {}

    for task_id in TASKS:
        try:
            scores[task_id] = run_task(ENV_URL, task_id)
        except Exception as e:
            print(f"❌ Task '{task_id}' failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*60}")
    print("🏁 FINAL RESULTS")
    print(f"{'='*60}")

    for task, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{task:8s}: [{bar}] {score:.4f}")

    avg = sum(scores.values()) / len(scores)
    print(f"{'AVERAGE':8s}: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()