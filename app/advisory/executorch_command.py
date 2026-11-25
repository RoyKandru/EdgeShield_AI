import json
import logging
import os
import select
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from app.advisory.base import AdvisoryProvider
from app.advisory.template import TemplateAdvisory
from app.advisory.validator import normalize_language, sanitize_event_for_llm, validate_advisory_output
from app.config import settings
from app.schemas import AdvisoryResponse, ThreatEvent

log = logging.getLogger("edgeshield.advisory.executorch")


class _PersistentExecuTorchDaemon:
    """Keeps bin/executorch_runner --daemon warm in memory so Llama-3.2-1B answers in ~2.5s."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._key: tuple[str, str, str, str] | None = None
        self.ready: bool = False
        self.load_ms: float | None = None
        self.runtime_name: str = ""
        self.last_error: str = ""
        self.last_metrics: dict[str, Any] = {}

    def _resolve_paths(self, command: str) -> tuple[list[str], str, str]:
        argv = shlex.split(command.strip())
        if not argv:
            raise ValueError("ExecuTorch runner command is empty")
        base_dir = Path(__file__).resolve().parents[2]
        cand = Path(argv[0])
        if not cand.is_file():
            rel = base_dir / argv[0].lstrip("./")
            if rel.is_file():
                argv[0] = str(rel)
            elif (base_dir / "bin" / "executorch_runner").is_file():
                argv[0] = str(base_dir / "bin" / "executorch_runner")
        model_path = settings.executor_model_path
        tok_path = settings.executor_tokenizer_path
        if not Path(model_path).exists() and (base_dir / "models" / "llama.pte").exists():
            model_path = str(base_dir / "models" / "llama.pte")
        if not Path(tok_path).exists() and (base_dir / "models" / "tokenizer.model").exists():
            tok_path = str(base_dir / "models" / "tokenizer.model")
        return argv, model_path, tok_path

    def probe_artifacts(self, command: str) -> dict[str, Any]:
        try:
            argv, model_path, tok_path = self._resolve_paths(command)
        except Exception as exc:
            return {"artifacts_ok": False, "error": str(exc)}
        bin_ok = Path(argv[0]).is_file() and os.access(argv[0], os.X_OK)
        model_ok = Path(model_path).exists()
        tok_ok = Path(tok_path).exists()
        return {
            "artifacts_ok": bin_ok and model_ok and tok_ok,
            "binary": argv[0],
            "binary_ok": bin_ok,
            "model_path": model_path,
            "resolved_model_path": os.path.realpath(model_path) if model_ok else "",
            "model_ok": model_ok,
            "tokenizer_path": tok_path,
            "tokenizer_ok": tok_ok,
            "daemon_alive": self._proc is not None and self._proc.poll() is None and self.ready,
            "load_ms": self.load_ms,
            "runtime": self.runtime_name or "executorch-xnnpack-kleidiai-arm64",
            "last_metrics": self.last_metrics,
        }

    def ensure_started(self, command: str) -> bool:
        with self._lock:
            return self._ensure_started_locked(command)

    def _ensure_started_locked(self, command: str) -> bool:
        argv, model_path, tok_path = self._resolve_paths(command)
        real_model = os.path.realpath(model_path)
        key = (argv[0], model_path, real_model, tok_path)
        if self._proc is not None and self._proc.poll() is None and self.ready and self._key == key:
            return True
        self._stop_locked()
        if not (Path(argv[0]).is_file() and Path(model_path).exists() and Path(tok_path).exists()):
            self.last_error = f"Missing binary/model/tokenizer: {key}"
            return False

        cmd = [argv[0], "--daemon", "--model", model_path, "--tokenizer", tok_path]
        env = os.environ.copy()
        env["EDGESHIELD_EXECUTOR_MODEL_PATH"] = model_path
        env["EDGESHIELD_EXECUTOR_TOKENIZER_PATH"] = tok_path
        log.info("Starting persistent ExecuTorch Llama daemon: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self._key = key
        # Wait for ready JSON line
        ready_line = self._proc.stdout.readline() if self._proc.stdout else ""
        if not ready_line:
            self.last_error = "ExecuTorch daemon exited before signaling readiness"
            self._stop_locked()
            return False
        try:
            info = json.loads(ready_line.strip())
            self.ready = bool(info.get("ready", False))
            self.load_ms = info.get("load_ms")
            self.runtime_name = str(info.get("runtime", "executorch-xnnpack-kleidiai-arm64"))
            if not self.ready:
                self.last_error = "ExecuTorch runner failed to load .pte model"
                self._stop_locked()
                return False
            log.info("ExecuTorch Llama daemon READY (load_ms=%s)", self.load_ms)
            return True
        except Exception as exc:
            self.last_error = f"Invalid readiness payload: {exc}"
            self._stop_locked()
            return False

    def _stop_locked(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin and self._proc.poll() is None:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._key = None
        self.ready = False
        self.load_ms = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def infer(self, command: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        with self._lock:
            if not self._ensure_started_locked(command):
                raise RuntimeError(self.last_error or "ExecuTorch daemon unavailable")
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            t0 = time.perf_counter()
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            ready_r, _, _ = select.select([self._proc.stdout], [], [], float(timeout))
            if not ready_r:
                self._stop_locked()
                raise TimeoutError(f"ExecuTorch Llama daemon timed out after {timeout}s")
            line = self._proc.stdout.readline()
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if not line:
                self._stop_locked()
                raise RuntimeError("ExecuTorch daemon closed stdout unexpectedly")
            data = json.loads(line.strip())
            if "error" in data:
                raise RuntimeError(f"ExecuTorch daemon error: {data['error']}")
            metrics = dict(data.get("metrics") or {})
            metrics["wall_ms"] = wall_ms
            metrics["daemon_warm"] = True
            data["metrics"] = metrics
            self.last_metrics = metrics
            return data


_DAEMON = _PersistentExecuTorchDaemon()


def warmup_executorch_daemon_async() -> None:
    """Warm up the persistent Llama-3.2-1B ExecuTorch daemon in the background at server boot."""
    if settings.executor.lower() != "command" or not settings.executor_command.strip():
        return
    if not settings.executor_daemon:
        return
    status = _DAEMON.probe_artifacts(settings.executor_command)
    if not status.get("artifacts_ok"):
        log.warning("Skipping ExecuTorch warmup: %s", status)
        return
    t = threading.Thread(
        target=_DAEMON.ensure_started,
        args=(settings.executor_command,),
        name="executorch-llama-warmup",
        daemon=True,
    )
    t.start()


def shutdown_executorch_daemon() -> None:
    _DAEMON.stop()


def restart_executorch_daemon(model_path: str | None = None) -> bool:
    """Stop and reload the persistent ExecuTorch Llama daemon for a newly selected .pte model."""
    if model_path:
        req_path = Path(model_path)
        if req_path.exists():
            cur_path = Path(settings.executor_model_path)
            try:
                same_target = cur_path.exists() and cur_path.resolve() == req_path.resolve()
            except OSError:
                same_target = False
            if not same_target:
                settings.executor_model_path = str(req_path)

    _DAEMON.stop()
    if settings.executor.lower() != "command" or not settings.executor_command.strip():
        return True
    if not settings.executor_daemon:
        return True
    status = _DAEMON.probe_artifacts(settings.executor_command)
    if not status.get("artifacts_ok"):
        log.warning("ExecuTorch daemon restart skipped (missing artifacts): %s", status)
        return False
    return _DAEMON.ensure_started(settings.executor_command)


def get_executorch_status() -> dict[str, Any]:
    if settings.executor.lower() != "command" or not settings.executor_command.strip():
        return {"configured": False, "artifacts_ok": False, "daemon_alive": False}
    info = _DAEMON.probe_artifacts(settings.executor_command)
    info["configured"] = True
    return info


class ExecuTorchCommandAdvisory(AdvisoryProvider):
    """On-device Llama-family ExecuTorch advisory provider (persistent daemon + one-shot fallback)."""

    def __init__(self, command: str, timeout: int = 60):
        if not command.strip():
            raise ValueError("EDGESHIELD_EXECUTOR_COMMAND is required for command mode")
        self.command = command
        self.timeout = timeout

    def explain(self, event: ThreatEvent, language: str = "en") -> AdvisoryResponse:
        lang = normalize_language(language)
        safe_event, injection_blocked = sanitize_event_for_llm(event)
        runner_lang = lang
        payload = {
            "task": "explain_security_event",
            "language": runner_lang,
            "instructions": (
                "Explain only the supplied verified evidence concisely for a non-expert user. "
                "Never contradict the detector risk verdict. Never advise clicking links or sharing OTP/PIN."
            ),
            "event": safe_event,
        }

        if settings.executor_daemon and "executorch_runner" in self.command:
            output = _DAEMON.infer(self.command, payload, timeout=self.timeout)
        else:
            argv = shlex.split(self.command)
            if not argv:
                raise RuntimeError("ExecuTorch runner command is empty")
            env = os.environ.copy()
            env.setdefault("EDGESHIELD_EXECUTOR_MODEL_PATH", settings.executor_model_path)
            env.setdefault("EDGESHIELD_EXECUTOR_TOKENIZER_PATH", settings.executor_tokenizer_path)
            result = subprocess.run(
                argv,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                shell=False,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ExecuTorch runner failed: {result.stderr[-1000:]}")
            raw_out = result.stdout.strip()
            s, e = raw_out.find("{"), raw_out.rfind("}")
            if s != -1 and e >= s:
                raw_out = raw_out[s : e + 1]
            output = json.loads(raw_out)

        raw_explanation = str(output.get("explanation", ""))
        raw_recs = [str(x) for x in output.get("recommendations", [])]
        metrics = dict(output.get("metrics") or {})
        metrics["prompt_injection_blocked"] = injection_blocked

        # Be 100% honest when benign fast-path skipped Llama
        if metrics.get("fast_path_benign"):
            return TemplateAdvisory().explain(
                event,
                language=lang,
                fallback_used=False,
                validation_notes=["benign_fast_path_no_llm_needed"],
                provider_name="fast-path-benign",
                metrics=metrics,
            )

        checked = validate_advisory_output(event, raw_explanation, raw_recs, language=lang)
        val_notes = list(checked["notes"])
        if injection_blocked:
            val_notes.append("prompt_injection_sanitized")

        if not metrics.get("model_used", True) or not checked["valid"]:
            if not metrics.get("model_used", True) and "llm_empty_output" not in val_notes:
                val_notes.append("llm_empty_output")
            log.warning("Llama advisory rejected/empty (%s); serving verdict-locked fallback", val_notes)
            return TemplateAdvisory().explain(
                event,
                language=lang,
                fallback_used=True,
                validation_notes=val_notes,
                provider_name="llama-executorch-validated-fallback",
                metrics=metrics,
            )

        tps = metrics.get("decode_tok_per_sec")
        inf_ms = metrics.get("inference_ms")
        lang_tag = {"hinglish": "HINGLISH", "hi": "HI"}.get(lang, "EN")
        note = f"Generated locally on Arm CPU by Llama-3.2-1B-Instruct via ExecuTorch (XNNPACK + KleidiAI) · Verdict-Locked Validated ({lang_tag})."
        if tps and inf_ms:
            note += f" [{inf_ms:.0f} ms, {tps:.1f} tok/s]"

        return AdvisoryResponse(
            event_id=event.event_id,
            provider="llama-executorch",
            language=lang,
            explanation=checked["explanation"],
            recommendations=checked["recommendations"],
            generated_locally=True,
            runtime_note=note,
            validated=True,
            fallback_used=False,
            prompt_injection_guarded=True,
            validation_notes=val_notes,
            metrics=metrics,
        )
