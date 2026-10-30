import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from cryptography.fernet import Fernet
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

request_records = Table(
    "api_request_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(32), nullable=False, index=True),
    Column("method", String(12), nullable=False),
    Column("path", String(1024), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("duration_ms", Float, nullable=False),
    Column("request_payload", LargeBinary(length=16_777_215), nullable=False),
    Column("response_payload", LargeBinary(length=16_777_215), nullable=False),
    Column("created_at", DateTime, nullable=False, index=True),
)

audit_events = Table(
    "api_audit_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(32), nullable=True, index=True),
    Column("event_type", String(100), nullable=False),
    Column("payload", LargeBinary(length=16_777_215), nullable=False),
    Column("occurred_at", DateTime, nullable=False, index=True),
)


class DataStore:
    def __init__(
        self,
        database_url: str,
        encryption_key: str,
        retention_days: int = 30,
        engine: Engine | None = None,
    ):
        if retention_days < 1:
            raise ValueError("DATA_RETENTION_DAYS must be at least 1")
        try:
            self._cipher = Fernet(encryption_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as error:
            raise ValueError("APP_DATA_ENCRYPTION_KEY must be a valid Fernet key") from error
        self._engine = engine or create_engine(database_url, pool_pre_ping=True)
        self._retention_days = retention_days

    def initialize(self) -> None:
        metadata.create_all(self._engine)

    def record_request(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_payload: bytes,
        response_payload: bytes,
        created_at: datetime | None = None,
    ) -> None:
        values = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "request_payload": self._encrypt(request_payload),
            "response_payload": self._encrypt(response_payload),
            "created_at": _naive_utc(created_at or datetime.now(timezone.utc)),
        }
        with self._engine.begin() as connection:
            connection.execute(insert(request_records).values(**values))

    def record_audit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        occurred_at = _parse_timestamp(payload.get("timestamp"))
        request_id = payload.get("request_id")
        encoded_payload = json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
        values = {
            "request_id": str(request_id) if request_id is not None else None,
            "event_type": event_type[:100],
            "payload": self._encrypt(encoded_payload),
            "occurred_at": _naive_utc(occurred_at),
        }
        with self._engine.begin() as connection:
            connection.execute(insert(audit_events).values(**values))

    def purge_expired(self, now: datetime | None = None) -> int:
        cutoff = _naive_utc(now or datetime.now(timezone.utc)) - timedelta(days=self._retention_days)
        with self._engine.begin() as connection:
            request_result = connection.execute(
                delete(request_records).where(request_records.c.created_at < cutoff)
            )
            audit_result = connection.execute(
                delete(audit_events).where(audit_events.c.occurred_at < cutoff)
            )
        return (request_result.rowcount or 0) + (audit_result.rowcount or 0)

    def _encrypt(self, payload: bytes) -> bytes:
        return self._cipher.encrypt(payload)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def get_data_store() -> DataStore | None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    encryption_key = os.getenv("APP_DATA_ENCRYPTION_KEY")
    if not encryption_key:
        raise RuntimeError("APP_DATA_ENCRYPTION_KEY is required when DATABASE_URL is configured")

    try:
        retention_days = int(os.getenv("DATA_RETENTION_DAYS", "30"))
    except ValueError as error:
        raise RuntimeError("DATA_RETENTION_DAYS must be a positive integer") from error
    if retention_days < 1:
        raise RuntimeError("DATA_RETENTION_DAYS must be a positive integer")

    return DataStore(database_url, encryption_key, retention_days)
