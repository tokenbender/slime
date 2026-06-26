"""Self-contained local-script harness for smoke tests.

This harness intentionally avoids external agent CLIs. It makes one
OpenAI-compatible request to slime's adapter so the rollout still contains
model-sampled tokens, then runs an optional local shell command in the sandbox.
"""

from __future__ import annotations

import os
import shlex
from textwrap import dedent

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, run_command


class LocalScriptHarness(BaseHarness):
    name = "local_script"

    command_env = "SLIME_AGENT_LOCAL_SCRIPT_COMMAND"
    max_tokens_env = "SLIME_AGENT_LOCAL_SCRIPT_MAX_TOKENS"

    async def install_cli(self, sb: Sandbox) -> None:
        await sb.exec("python3 --version", user="root", check=True, timeout=30)

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        await sb.exec("mkdir -p /home/agent/.local_script && chown -R agent:agent /home/agent/.local_script", user="root", check=True, timeout=30)

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        script = dedent(
            """
            import json
            import os
            import subprocess
            import sys
            import urllib.request

            base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
            token = os.environ["OPENAI_API_KEY"]
            model = os.environ.get("OPENAI_MODEL", "slime-actor")
            prompt = os.environ.get("LOCAL_SCRIPT_PROMPT", "Say OK.")
            max_tokens = int(os.environ.get("LOCAL_SCRIPT_MAX_TOKENS", "32"))

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": float(os.environ.get("LOCAL_SCRIPT_TEMPERATURE", "0")),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(
                base_url + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"].get("content") or ""
            print(content)

            command = os.environ.get("LOCAL_SCRIPT_COMMAND", "").strip()
            if command:
                result = subprocess.run(command, shell=True, cwd=os.getcwd())
                sys.exit(result.returncode)
            """
        ).strip() + "\n"
        agent_path = f"{ctx.workdir}/.harness/local_script_agent.py"
        await sb.exec(f"mkdir -p {ctx.workdir}/.harness && chown agent:agent {ctx.workdir}/.harness", user="root", check=True, timeout=30)
        await sb.write_file(agent_path, script, user="agent")

        command = os.environ.get(self.command_env, "").strip()
        max_tokens = os.environ.get(self.max_tokens_env, "32")
        env = {
            "OPENAI_API_KEY": ctx.session_id,
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
            "OPENAI_MODEL": ctx.model_label,
            "LOCAL_SCRIPT_PROMPT": prompt,
            "LOCAL_SCRIPT_MAX_TOKENS": str(max_tokens),
        }
        if command:
            env["LOCAL_SCRIPT_COMMAND"] = command
        return await run_command(
            sb,
            workdir=ctx.workdir,
            start_cmd=f"python3 {shlex.quote(agent_path)}",
            env=env,
            time_budget_sec=time_budget_sec,
        )
