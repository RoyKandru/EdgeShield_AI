from app.config import settings
from app.advisory.template import TemplateAdvisory
from app.advisory.executorch_command import ExecuTorchCommandAdvisory


def get_advisory_provider():
    if settings.executor.lower() == "command":
        return ExecuTorchCommandAdvisory(
            command=settings.executor_command,
            timeout=settings.executor_timeout,
        )
    return TemplateAdvisory()
