import asyncio
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.adaptive.registry import evaluate_candidate
from app.adaptive.schemas import CandidatePredictionRequest, FeedbackRequest, NetworkObservation
from app.adaptive.service import AdaptiveEngine
from app.advisory.executorch_command import (
    get_executorch_status,
    restart_executorch_daemon,
    shutdown_executorch_daemon,
    warmup_executorch_daemon_async,
)
from app.advisory.factory import get_advisory_provider
from app.auth import is_auth_enabled, require_api_token
from app.config import settings
from app.db import EventStore
from app.detectors.live_monitor import LiveNetworkMonitor
from app.detectors.network import analyse_flows, analyse_pcap
from app.detectors.sms import analyse_sms
from app.detectors.url import analyse_url
from app.platform.monitor import collect_system_metrics
from app.risk import CrossChannelCorrelator
from app.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    AnalyzeNetworkFlowRequest,
    AnalyzeSMSRequest,
    AnalyzeURLRequest,
    ContainmentRequest,
    LinkSandboxRequest,
    SelectModelRequest,
    SystemStatus,
    ThreatEvent,
)
from app.sci.closed_loop import SCIClosedLoopController
from app.detectors.link_sandbox import inspect_url_sandbox
from app.sci.containment import ContainmentEngine, infer_target_from_event
from app.sci.engine import SCIEngine
from app.sci.policy import get_policy_manager
from app.sci.schemas import (
    SCIClosedLoopSubmitRequest,
    SCIIncidentRequest,
    SCIProposalDeployRequest,
    SCIRollbackRequest,
)

log = logging.getLogger("edgeshield.api")

store = EventStore(settings.db_path)
adaptive = AdaptiveEngine(settings.db_path, "./models/adaptive")
sci = SCIEngine(settings.db_path)
containment = ContainmentEngine(settings.db_path, sci)
policy_mgr = get_policy_manager()
sci_loop = SCIClosedLoopController(settings.db_path, sci, policy_mgr)
correlator = CrossChannelCorrelator()


def persist(event: ThreatEvent):
    payload = event.model_dump(mode="json")
    store.save_event(payload)
    try:
        sci_loop.record_decision((payload.get("metadata") or {}).get("sci_policy_version"))
    except Exception:
        pass
    corr = correlator.ingest(payload)
    if corr is not None:
        store.save_correlation(corr)
    return payload


def persist_many(events: list[ThreatEvent]) -> list[dict]:
    payloads = [e.model_dump(mode="json") for e in events]
    store.save_events_batch(payloads)
    for p in payloads[:50]:
        if p.get("risk_score", 0) >= 30:
            corr = correlator.ingest(p)
            if corr is not None:
                store.save_correlation(corr)
    return payloads


live_monitor = LiveNetworkMonitor(
    on_event=persist,
    window_s=settings.live_monitor_window_s,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup_executorch_daemon_async()
    if settings.live_monitor_enabled:
        live_monitor.start(settings.live_monitor_interface)
    yield
    live_monitor.stop()
    shutdown_executorch_daemon()


app = FastAPI(
    title="EdgeShield AI",
    description="On-device Arm RK3588J security assistant: live network IDS, phishing URL & scam SMS detection, 9-stage closed-loop SCI self-correction, RK3588 3-core NPU adapter, and Llama-3.2-1B ExecuTorch advisory.",
    version="1.4.1-arm64-sci",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse("app/static/index.html")


@app.get("/api/system/status", response_model=SystemStatus)
def system_status():
    et = get_executorch_status()
    llama_ready = bool(et.get("configured") and et.get("artifacts_ok"))
    mon = live_monitor.status()
    pol_ver = policy_mgr.get_active().get("version", "v1")
    return SystemStatus(
        status="ok",
        application="EdgeShield AI",
        offline_mode=True,
        llama_provider="llama-executorch" if llama_ready else "deterministic-template",
        llama_ready=llama_ready,
        database="sqlite-wal",
        optional_network_capture=True,
        llama_model="Llama-3.2-1B-Instruct (XNNPACK)" if llama_ready else "",
        llama_runtime=str(et.get("runtime", "")) if llama_ready else "",
        llama_load_ms=et.get("load_ms"),
        live_monitor_running=bool(mon.get("running")),
        live_monitor_interface=str(mon.get("interface", "")),
        api_auth_enabled=is_auth_enabled(),
        detector_versions={
            "url": "url-hybrid-v3-npu-dual (RK3588 NPU_CORE_1 Prasad 2026 MLP+CharCNN)",
            "sms": "sms-hybrid-v3-npu-bigru (RK3588 NPU_CORE_2 BiGRU+MaskedPool+WBCE)",
            "network": "network-hybrid-v3-npu-1dcnn (RK3588 NPU_CORE_0 + live-wire-dpi)",
            "advisory": "llama-3.2-1b-executorch" if llama_ready else "deterministic-template",
            "sci_policy": pol_ver,
            "rknn_npu_adapter": "vv1-3core-all-npu-live",
        },
    )


@app.get("/api/system/metrics")
def system_metrics():
    return collect_system_metrics()


@app.get("/api/stats")
def stats():
    base = store.stats()
    base["live_monitor"] = live_monitor.status()
    base["llama"] = get_executorch_status()
    base["sci_policy_version"] = policy_mgr.get_active().get("version", "v1")
    return base


@app.get("/api/events")
def events(limit: int = 100):
    return store.list_events(max(1, min(limit, 500)))


@app.get("/api/events/{event_id}")
def event_detail(event_id: str):
    item = store.get_event(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="Event not found")
    return item


@app.get("/api/correlations")
def list_correlations(limit: int = 50):
    return {"correlations": store.list_correlations(max(1, min(limit, 200)))}


@app.post("/api/url/analyze")
def url_analyze(request: AnalyzeURLRequest):
    try:
        return persist(analyse_url(request.url))
    except Exception as exc:
        sci.record_incident("url", "URL analysis pipeline exception",
            {"exception_type": type(exc).__name__, "message": str(exc)[:300]},
            "mark_uncertain", {"checks": [], "independent": False}, actor="url-api")
        raise HTTPException(status_code=500, detail="URL analysis failed; an SCI incident was recorded")


@app.post("/api/sms/analyze")
def sms_analyze(request: AnalyzeSMSRequest):
    try:
        return persist(analyse_sms(request.message))
    except Exception as exc:
        sci.record_incident("sms", "SMS analysis pipeline exception",
            {"exception_type": type(exc).__name__, "message": str(exc)[:300]},
            "mark_uncertain", {"checks": [], "independent": False}, actor="sms-api")
        raise HTTPException(status_code=500, detail="SMS analysis failed; an SCI incident was recorded")


@app.post("/api/analyze/network")
@app.post("/api/network/analyze")
def network_analyze_flow(request: AnalyzeNetworkFlowRequest):
    try:
        flow_dict = request.model_dump()
        if flow_dict.get("syn_packets", 0) and not flow_dict.get("syn_count", 0):
            flow_dict["syn_count"] = flow_dict["syn_packets"]
        ev = analyse_flows([flow_dict])[0]
        return persist(ev)
    except Exception as exc:
        sci.record_incident("network", "Network flow analysis exception",
            {"exception_type": type(exc).__name__, "message": str(exc)[:300]},
            "mark_uncertain", {"checks": [], "independent": False}, actor="network-json-api")
        raise HTTPException(status_code=500, detail=f"Network analysis failed: {exc}")


@app.post("/api/network/analyze-pcap")
async def network_analyze_pcap(file: UploadFile = File(...)):
    suffix = Path(file.filename or "capture.pcap").suffix.lower()
    if suffix not in {".pcap", ".pcapng", ".cap"}:
        raise HTTPException(status_code=400, detail="Upload a .pcap, .pcapng or .cap file")
    max_bytes = 50 * 1024 * 1024
    temp_path = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="PCAP exceeds 50 MB limit")
                tmp.write(chunk)
        detections = await asyncio.to_thread(analyse_pcap, temp_path)
        persisted = await asyncio.to_thread(persist_many, detections)
        flagged = sum(1 for e in persisted if e.get("prediction") != "no_configured_anomaly")
        return {
            "flows_analysed": len(persisted),
            "flagged_flows": flagged,
            "events": persisted,
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        sci.record_incident("network", "Network capture analysis unavailable",
            {"exception_type": type(exc).__name__, "message": str(exc)[:300]},
            "mark_uncertain", {"checks": [], "independent": False}, actor="network-pcap-api")
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        sci.record_incident("network", "Network capture processing exception",
            {"exception_type": type(exc).__name__, "message": str(exc)[:300]},
            "mark_uncertain", {"checks": [], "independent": False}, actor="network-pcap-api")
        raise HTTPException(status_code=400, detail=f"Could not analyse capture: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/api/network/monitor/status")
def network_monitor_status():
    return live_monitor.status()


@app.post("/api/network/monitor/start", dependencies=[Depends(require_api_token)])
def network_monitor_start(interface: str = "auto"):
    return live_monitor.start(interface)


@app.post("/api/network/monitor/stop", dependencies=[Depends(require_api_token)])
def network_monitor_stop():
    return live_monitor.stop()


@app.get("/api/network/devices")
def network_devices():
    return {"devices": live_monitor.list_devices()}


def _friendly_pte_label(filename: str) -> str:
    if filename == "llama_3_2_1b_xnnpack_ctx512.pte":
        return "Llama-3.2-1B (ctx=512)"
    if filename == "llama_3_2_1b_xnnpack.pte":
        return "Llama-3.2-1B (ctx=128)"
    stem = Path(filename).stem
    return stem.replace("_", " ").strip() or filename


def _discover_network_interfaces(active_iface: str) -> list[dict]:
    net_root = Path("/sys/class/net")
    items: list[dict] = []
    if net_root.is_dir():
        for entry in net_root.iterdir():
            name = entry.name
            operstate = "unknown"
            try:
                raw_state = (entry / "operstate").read_text(encoding="utf-8").strip().lower()
                if raw_state:
                    operstate = raw_state
            except OSError:
                pass
            items.append({
                "name": name,
                "operstate": operstate,
                "is_active_monitor": bool(active_iface and name == active_iface),
            })

    def _sort_key(it: dict) -> tuple[int, int, str]:
        name = it["name"]
        state = it["operstate"]
        is_virtual = name == "lo" or name.startswith(("docker", "br-", "veth", "virbr", "can", "tun", "tap"))
        if it["is_active_monitor"]:
            tier = 0
        elif not is_virtual and state == "up":
            tier = 1
        elif not is_virtual:
            tier = 2
        elif name.startswith("docker"):
            tier = 3
        elif name.startswith("can"):
            tier = 4
        elif name == "lo":
            tier = 6
        else:
            tier = 5
        state_rank = 0 if state == "up" else (1 if state == "unknown" else 2)
        return (tier, state_rank, name)

    items.sort(key=_sort_key)
    return items


def _discover_llm_models() -> list[dict]:
    model_dir = Path(settings.model_dir)
    alias_path = model_dir / "llama.pte"
    active_resolved: Path | None = None
    if alias_path.exists():
        try:
            active_resolved = alias_path.resolve()
        except OSError:
            active_resolved = None
    if active_resolved is None and settings.executor_model_path:
        try:
            ep = Path(settings.executor_model_path)
            if ep.exists():
                active_resolved = ep.resolve()
        except OSError:
            pass

    models: list[dict] = []
    if model_dir.is_dir():
        for p in model_dir.glob("*.pte"):
            if p.name == "llama.pte":
                continue
            if not p.exists():
                continue
            try:
                size = p.stat().st_size
                resolved = p.resolve()
            except OSError:
                continue
            is_active = bool(active_resolved is not None and resolved == active_resolved)
            models.append({
                "id": p.name,
                "label": _friendly_pte_label(p.name),
                "path": str(p),
                "size_mb": round(size / 1048576.0, 1),
                "max_context_len": 512 if "ctx512" in p.name else 128,
                "active": is_active,
            })
    models.sort(key=lambda m: (not m["active"], -m["max_context_len"], m["id"]))
    return models


@app.get("/api/hardware/runtime-options")
def hardware_runtime_options():
    mon = live_monitor.status()
    monitor_running = bool(mon.get("running"))
    active_interface = str(mon.get("interface", "") or "")
    return {
        "interfaces": _discover_network_interfaces(active_interface),
        "monitor_running": monitor_running,
        "active_interface": active_interface,
        "llm_models": _discover_llm_models(),
        "languages": [
            {"code": "en", "label": "English"},
            {"code": "hi", "label": "हिन्दी (Hindi)"},
            {"code": "hinglish", "label": "Hinglish"},
        ],
    }


@app.post("/api/advisory/select-model", dependencies=[Depends(require_api_token)])
def advisory_select_model(request: SelectModelRequest):
    model_id = request.model_id.strip()
    if (
        not model_id
        or "/" in model_id
        or "\\" in model_id
        or ".." in model_id
        or Path(model_id).name != model_id
        or not model_id.endswith(".pte")
        or model_id == "llama.pte"
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid model_id: must be a safe .pte filename under the model directory",
        )

    model_dir = Path(settings.model_dir)
    target_path = model_dir / model_id
    try:
        if target_path.parent.resolve() != model_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid model_id path")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Model directory error: {exc}")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    symlink_path = model_dir / "llama.pte"
    tmp_link = model_dir / f".llama.pte.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(model_id, tmp_link)
        os.replace(tmp_link, symlink_path)
    except OSError as exc:
        if tmp_link.exists() or tmp_link.is_symlink():
            try:
                tmp_link.unlink()
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to update llama.pte symlink: {exc}")

    daemon_restarted = restart_executorch_daemon(str(target_path))
    if not daemon_restarted:
        raise HTTPException(
            status_code=503,
            detail=f"Updated llama.pte -> {model_id}, but ExecuTorch daemon failed to reload",
        )

    return {
        "status": "ok",
        "active_model": model_id,
        "label": _friendly_pte_label(model_id),
        "daemon_restarted": True,
    }


@app.post("/api/advisory/explain", response_model=AdvisoryResponse)
def explain(request: AdvisoryRequest):
    payload = store.get_event(request.event_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Threat event not found")
    event = ThreatEvent.model_validate(payload)
    try:
        provider = get_advisory_provider()
        return provider.explain(event, language=request.language)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Advisory provider error: {exc}")


# --- RK3588 3-Core NPU Adapter Probe Endpoints (merged from v1) ---

@app.get("/api/hardware/rknn/status")
def rknn_status():
    """Report actual RK3588 NPU 3-core readiness per detector (separate from ExecuTorch Llama Arm CPU)."""
    try:
        from app.edge_models.service import get_edge_inference_service
        return get_edge_inference_service().status()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RKNN service unavailable: {type(exc).__name__}: {exc}")


@app.post("/api/hardware/rknn/network")
def rknn_network(request: dict):
    try:
        from app.edge_models.service import get_edge_inference_service
        result = get_edge_inference_service().infer_network(request.get("features", request))
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RKNN network inference unavailable: {exc}")


@app.post("/api/hardware/rknn/url")
def rknn_url(request: AnalyzeURLRequest):
    try:
        from app.edge_models.service import get_edge_inference_service
        result = get_edge_inference_service().infer_url(request.url)
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RKNN URL inference unavailable: {exc}")


@app.post("/api/hardware/rknn/sms")
def rknn_sms(request: AnalyzeSMSRequest):
    try:
        from app.edge_models.service import get_edge_inference_service
        result = get_edge_inference_service().infer_sms(request.message)
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RKNN SMS inference unavailable: {exc}")


# --- Adaptive Learning Endpoints ---

@app.post("/api/adaptive/feedback", dependencies=[Depends(require_api_token)])
def submit_feedback(request: FeedbackRequest):
    event = store.get_event(request.event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Threat event not found")
    raw_source = event.get("source", "unknown")
    source_norm = "sms" if "sms" in raw_source else "url" if "url" in raw_source else "network"
    content = event.get("raw_input_preview", "")
    item = {
        "event_id": request.event_id, "verdict": request.verdict,
        "verified": request.verified, "reviewer": request.reviewer,
        "notes": request.notes, "source": source_norm, "content": content,
        "label": request.label,
    }
    try:
        return adaptive.accept_feedback(item)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Feedback could not be processed: {exc}")


@app.post("/api/adaptive/network/observe", dependencies=[Depends(require_api_token)])
def observe_network(request: NetworkObservation):
    return {
        "observations": adaptive.observe_network(request.metrics),
        "note": "EWMA baselines are local and adaptive; drift flags are advisory, not proof of an attack.",
    }


@app.post("/api/adaptive/candidate/predict")
def predict_with_candidate(request: CandidatePredictionRequest):
    return adaptive.learner.predict(request.text)


@app.post("/api/adaptive/candidate/evaluate", dependencies=[Depends(require_api_token)])
def evaluate_adaptive_candidate(metrics: dict):
    result = evaluate_candidate(metrics)
    if result["eligible"]:
        model_id = adaptive.store.register_model(
            pipeline="text", version="candidate-1", status="eligible_for_human_approval",
            metrics=metrics, artifact_path=str(adaptive.learner.path))
        result["model_id"] = model_id
        result["status"] = "eligible_for_human_approval"
    return result


@app.get("/api/adaptive/status")
def adaptive_status():
    return adaptive.status()


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "v1", "auth_enabled": is_auth_enabled()}


# --- SCI Ledger + 9-Stage Closed-Loop Self-Learning & Self-Correction Endpoints ---

@app.post("/api/sci/incidents", dependencies=[Depends(require_api_token)])
def sci_record_incident(request: SCIIncidentRequest):
    try:
        return sci.record_incident(
            component=request.component, symptom=request.symptom,
            evidence=request.evidence, correction_action=request.correction_action,
            verification=request.verification, actor=request.actor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/sci/incidents")
def sci_list_incidents(limit: int = 50):
    return {"incidents": sci.list_incidents(limit)}


@app.get("/api/sci/ledger/verify")
def sci_verify_ledger():
    return sci.verify_ledger()


@app.get("/api/sci/capabilities")
def sci_capabilities():
    return {
        "components_monitored": ["network", "url", "sms", "model", "soc", "runtime"],
        "input_pipelines": ["network", "url", "sms"],
        "stages": [
            "threat_detection", "evidence_logging", "outcome_verification",
            "error_diagnosis", "correction_planner", "correction_engine",
            "validation_engine", "controlled_deployment_and_rollback", "continuous_monitoring",
        ],
        "allowlisted_actions": sci.ALLOWED_ACTIONS,
        "external_sensors": False,
        "automatic_physical_repair": False,
        "automatic_shell_execution": False,
        "llm_policy_modification": False,
        "proof": "SHA-256 hash-linked local audit records + automated golden regression validation before deployment",
    }


@app.get("/api/sci/pipeline/status")
def sci_pipeline_status():
    return sci_loop.pipeline_status()


@app.post("/api/sci/closed-loop/submit", dependencies=[Depends(require_api_token)])
def sci_closed_loop_submit(request: SCIClosedLoopSubmitRequest):
    event = store.get_event(request.event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Threat event not found")
    try:
        return sci_loop.submit_error_cycle(
            event=event,
            claimed_verdict=request.claimed_verdict,
            claimed_label=request.claimed_label,
            reviewer=request.reviewer,
            independent_checks=request.independent_checks,
            notes=request.notes,
            auto_deploy_if_validated=request.auto_deploy_if_validated,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/sci/proposals")
def sci_list_proposals(limit: int = 50):
    return {"proposals": sci_loop.list_proposals(max(1, min(limit, 200)))}


@app.post("/api/sci/proposals/{proposal_id}/validate", dependencies=[Depends(require_api_token)])
def sci_validate_proposal(proposal_id: str):
    try:
        return sci_loop.validate_proposal(proposal_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/sci/proposals/{proposal_id}/deploy", dependencies=[Depends(require_api_token)])
def sci_deploy_proposal(proposal_id: str, request: SCIProposalDeployRequest):
    try:
        return sci_loop.deploy_proposal(proposal_id, actor=request.actor)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/sci/rollback", dependencies=[Depends(require_api_token)])
def sci_rollback(request: SCIRollbackRequest):
    try:
        return sci_loop.rollback(
            target_version=request.target_version,
            actor=request.actor,
            reason=request.reason,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/sci/intel")
def sci_intel():
    active = policy_mgr.get_active()
    return {
        "policy_version": active.get("version"),
        "sha256": active.get("sha256"),
        "verified_allow_domains": active.get("verified_allow_domains", []),
        "verified_deny_domains": active.get("verified_deny_domains", []),
        "verified_deny_indicators": active.get("verified_deny_indicators", []),
        "verified_benign_phrases": active.get("verified_benign_phrases", []),
        "rule_weights": active.get("rule_weights", {}),
        "thresholds": active.get("thresholds", {}),
    }


@app.get("/api/sci/retrain-pack")
def sci_retrain_pack():
    return sci_loop.export_retrain_pack()


@app.get("/api/sci/monitor")
def sci_monitor():
    return {"versions": sci_loop.monitor_status()}


# --- Reversible Graduated Containment Ladder Endpoints ---

@app.get("/api/containment/status")
def containment_status():
    return containment.status()


@app.post("/api/containment/contain", dependencies=[Depends(require_api_token)])
def containment_apply(request: ContainmentRequest):
    action = request.action.strip()
    target = request.target.strip()
    risk_score = 80
    if request.event_id:
        ev = store.get_event(request.event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Threat event not found")
        risk_score = int(ev.get("risk_score", 80))
        if not action or not target:
            try:
                action, target = infer_target_from_event(ev)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
    if not action or not target:
        raise HTTPException(status_code=400, detail="Provide either event_id or explicit action + target")
    try:
        return containment.contain(
            action=action,
            target=target,
            event_id=request.event_id,
            risk_score=risk_score,
            ttl_seconds=request.ttl_seconds,
            actor=request.actor,
            reason=request.reason,
            force=request.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/containment/undo/{action_id}", dependencies=[Depends(require_api_token)])
def containment_undo(action_id: str, actor: str = "soc-operator"):
    try:
        return containment.undo(action_id, actor=actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --- Opt-in Isolated Link Sandbox Endpoint (OFF by default in strict offline mode) ---

@app.post("/api/url/sandbox")
def url_sandbox_inspect(request: LinkSandboxRequest):
    try:
        return inspect_url_sandbox(
            request.url,
            html_content=request.html_content,
            allow_network=request.allow_network,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Sandbox inspection error: {exc}")
