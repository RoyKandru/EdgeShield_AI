"""Optional constant-time API token authentication for EdgeShield control-plane endpoints.

Set EDGESHIELD_API_TOKEN=<secret> to enforce authentication on mutating SCI,
Live Network Monitor, and Adaptive Learning endpoints. When unset/empty,
local trusted-LAN mode stays frictionless and /api/health reports auth_enabled=false.
"""
from __future__ import annotations

import hmac
from typing import Any

from fastapi import Header, HTTPException, status

from app.config import settings


def is_auth_enabled() -> bool:
    return bool((settings.api_token or "").strip())


def _extract_candidate(x_token: str | None, authorization: str | None) -> str:
    if x_token and x_token.strip():
        return x_token.strip()
    if authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return ""


def require_api_token(
    x_edgeshield_token: str | None = Header(default=None, alias="X-EdgeShield-Token"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    expected = (settings.api_token or "").strip()
    if not expected:
        return {"auth_enabled": False, "authenticated": True}
    provided = _extract_candidate(x_edgeshield_token, authorization)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-EdgeShield-Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"auth_enabled": True, "authenticated": True}
