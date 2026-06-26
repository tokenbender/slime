"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


def sandbox_backend() -> str:
    """Selected sandbox backend for agent rollouts."""
    return _getenv("SLIME_AGENT_SANDBOX_BACKEND", "SWE_SANDBOX_BACKEND", default="e2b").strip().lower()


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")

    default_lifetime_sec = 3600
    default_rpc_retries = 3
    # With retries=3 the sleep budget is 3s, which handles common E2B h2 reset
    # / SSL / pool-timeout flaps without stalling rollout steps for too long.
    rpc_backoff_base_sec = 1.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @staticmethod
    def _is_transient_rpc_error(e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory):
        """Run coro_factory() with retries for transient E2B RPC failures."""
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    backoff = self.rpc_backoff_base_sec * (2**attempt)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  # type: ignore

        md = {self.image_metadata_key: self.image}
        self._sb = await AsyncSandbox.create(timeout=self.timeout, metadata=md)
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


class LocalDockerSandbox:
    """Local Docker implementation of the agent sandbox contract.

    The dataset ``image`` field is interpreted as a Docker image available to the
    host. Each sandbox is an isolated throwaway container, which keeps the
    coding-agent example usable without E2B or an external sandbox gateway.
    """

    docker_bin_env = "SLIME_AGENT_DOCKER_BIN"
    docker_run_args_env = "SLIME_AGENT_DOCKER_RUN_ARGS"
    keep_container_env = "SLIME_AGENT_DOCKER_KEEP_CONTAINER"

    def __init__(self, image: str, *, timeout: int | None = None) -> None:
        self.image = image
        self.timeout = timeout
        self.docker_bin = os.environ.get(self.docker_bin_env, "docker")
        self.sandbox_id = ""
        self._container = ""

    async def __aenter__(self) -> LocalDockerSandbox:
        name = f"slime-agent-{uuid.uuid4().hex[:12]}"
        args = [
            self.docker_bin,
            "run",
            "-d",
            "--rm",
            "--init",
            "--name",
            name,
            "--add-host=host.docker.internal:host-gateway",
            *shlex.split(os.environ.get(self.docker_run_args_env, "")),
            "--entrypoint",
            "/bin/bash",
            self.image,
            "-lc",
            "sleep infinity",
        ]
        code, out, err = await _run_host(args, timeout=120)
        if code != 0:
            raise RuntimeError(f"docker run failed for {self.image!r}: {err[:400] or out[:400]}")
        self._container = out.strip()
        self.sandbox_id = self._container
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._container or os.environ.get(self.keep_container_env):
            return
        code, _, err = await _run_host([self.docker_bin, "rm", "-f", self._container], timeout=60)
        if code != 0:
            logger.warning("[agent.sandbox] docker rm %s failed: %s", self._container[:12], err[:200])

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        if not self._container:
            raise RuntimeError("LocalDockerSandbox has not been entered")
        run_cmd = cmd
        wait_timeout = None
        if timeout and timeout > 0:
            run_cmd = f"timeout --kill-after=5s {int(timeout)}s bash -lc {shlex.quote(cmd)}"
            wait_timeout = int(timeout) + 10
        args = [self.docker_bin, "exec"]
        for k, v in (env or {}).items():
            args.extend(["--env", f"{k}={v}"])
        if user:
            args.extend(["--user", user])
        args.extend([self._container, "bash", "-lc", run_cmd])
        code, out, err = await _run_host(args, timeout=wait_timeout)
        if check and code != 0:
            raise RuntimeError(f"docker exec failed (exit={code}): {cmd[:120]}\n{err[:400]}")
        return code, out, err

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if not self._container:
            raise RuntimeError("LocalDockerSandbox has not been entered")
        await self.exec(f"mkdir -p {shlex.quote(str(Path(sandbox_path).parent))}", user="root", check=True, timeout=30)
        host_path = content if isinstance(content, Path) else await _write_temp(content)
        remove_tmp = not isinstance(content, Path)
        try:
            code, out, err = await _run_host(
                [self.docker_bin, "cp", str(host_path), f"{self._container}:{sandbox_path}"], timeout=600
            )
            if code != 0:
                raise RuntimeError(f"docker cp failed for {sandbox_path}: {err[:400] or out[:400]}")
        finally:
            if remove_tmp:
                try:
                    os.unlink(host_path)
                except OSError:
                    pass
        if user and user != "root":
            q = shlex.quote(sandbox_path)
            await self.exec(f"chown {shlex.quote(user)}:{shlex.quote(user)} {q}", user="root", check=False, timeout=30)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        code, out, _ = await self.exec(f"cat {shlex.quote(sandbox_path)}", user=user, check=False, timeout=120)
        return out if code == 0 else ""


async def _write_temp(content: str | bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="slime-agent-docker-")
    mode = "wb" if isinstance(content, bytes) else "w"
    with os.fdopen(fd, mode) as f:
        f.write(content)
    return path


async def _run_host(args: list[str], *, timeout: int | None = None) -> ExecResult:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        return 124, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent",
        user="root",
        check=True,
        timeout=60,
    )
