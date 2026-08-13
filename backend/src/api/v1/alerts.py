from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.api.dependencies import AlertDep
from src.api.schemas.alerts import AlertAcknowledge, AlertResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/active", response_model=list[AlertResponse])
async def get_active_alerts(
    service: AlertDep,
    region: str | None = Query(default=None),
    severity: str | None = Query(default=None),
) -> list[AlertResponse]:
    """Return all unacknowledged, non-expired alerts."""
    return await service.get_active_alerts(region=region, severity=severity)


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    body: AlertAcknowledge,
    service: AlertDep,
) -> AlertResponse:
    """Acknowledge an alert by its ID."""
    return await service.acknowledge(alert_id=alert_id, by=body.acknowledged_by)


@router.websocket("/stream")
async def alert_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint that pushes new alerts in real-time.

    Clients connect and receive JSON-serialized AlertResponse objects
    whenever a new alert is created. Uses an in-process asyncio queue
    as the broadcast mechanism. In production, this would be backed
    by Redis Pub/Sub.
    """
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()

    subscribers = websocket.app.state.alert_subscribers
    subscribers.append(queue)

    try:
        while True:
            message = await queue.get()
            await websocket.send_text(message)
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.remove(queue)
