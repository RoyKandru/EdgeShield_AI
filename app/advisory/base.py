from abc import ABC, abstractmethod
from app.schemas import AdvisoryResponse, ThreatEvent


class AdvisoryProvider(ABC):
    @abstractmethod
    def explain(self, event: ThreatEvent) -> AdvisoryResponse:
        raise NotImplementedError
