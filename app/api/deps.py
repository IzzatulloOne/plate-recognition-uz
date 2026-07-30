"""Общие зависимости: доступ к состоянию приложения и проверка API-ключа."""

from __future__ import annotations

from fastapi import HTTPException, Request, WebSocket, status

from app.config import settings


def get_state(request: Request):
    return request.app.state.anpr


def get_ws_state(websocket: WebSocket):
    return websocket.app.state.anpr


async def require_api_key(request: Request) -> None:
    if not settings.api_key:
        return
    key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "неверный или отсутствующий X-API-Key")


def check_ws_api_key(websocket: WebSocket) -> bool:
    if not settings.api_key:
        return True
    key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    return key == settings.api_key
