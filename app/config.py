from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BIN = _ROOT / "bin" / "executorch_runner"
_DEFAULT_MODEL = _ROOT / "models" / "llama.pte"
_HAS_NATIVE_LLAMA = _DEFAULT_BIN.is_file() and _DEFAULT_MODEL.exists()


class Settings(BaseSettings):
    model_config = ConfigDict(env_prefix="EDGESHIELD_", protected_namespaces=())

    app_name: str = "EdgeShield AI"
    db_path: str = "./data/edgeshield.db"
    executor: str = "command" if _HAS_NATIVE_LLAMA else "template"  # template | command
    executor_command: str = str(_DEFAULT_BIN) if _HAS_NATIVE_LLAMA else ""
    executor_timeout: int = 30
    executor_daemon: bool = True
    executor_model_path: str = str(_DEFAULT_MODEL)
    executor_tokenizer_path: str = str(_ROOT / "models" / "tokenizer.model")
    ml_detectors: bool = True
    model_dir: str = str(_ROOT / "models")
    live_monitor_enabled: bool = True
    live_monitor_interface: str = "auto"
    live_monitor_window_s: float = 2.0
    api_token: str = ""
    containment_dry_run: bool = True
    containment_default_ttl_s: int = 3600
    protected_hosts: str = ""
    sandbox_enabled: bool = False


settings = Settings()
