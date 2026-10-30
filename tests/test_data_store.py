import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, select

from app.data_store import DataStore, audit_events, request_records


def _store(retention_days=30):
    engine = create_engine("sqlite:///:memory:")
    store = DataStore(
        "sqlite:///:memory:",
        Fernet.generate_key().decode("ascii"),
        retention_days=retention_days,
        engine=engine,
    )
    store.initialize()
    return store, engine


def test_request_and_audit_payloads_are_encrypted():
    store, engine = _store()
    request_payload = b'{"patient_nbr":123,"diagnosis":"private"}'
    response_payload = b'{"prediction":1,"probability":0.8}'
    audit_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": "request-1",
        "patient_nbr": 123,
    }

    store.record_request(
        request_id="request-1",
        method="POST",
        path="/predict",
        status_code=200,
        duration_ms=12.5,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    store.record_audit_event("prediction_completed", audit_payload)

    with engine.connect() as connection:
        request = connection.execute(select(request_records)).mappings().one()
        audit = connection.execute(select(audit_events)).mappings().one()

    assert request["request_payload"] != request_payload
    assert request["response_payload"] != response_payload
    assert store._cipher.decrypt(request["request_payload"]) == request_payload
    assert store._cipher.decrypt(request["response_payload"]) == response_payload
    assert audit["payload"] != json.dumps(audit_payload).encode()
    assert json.loads(store._cipher.decrypt(audit["payload"])) == audit_payload
    assert audit["request_id"] == "request-1"


def test_retention_purges_records_older_than_configured_days():
    store, engine = _store(retention_days=30)
    old_date = datetime.now(timezone.utc) - timedelta(days=31)
    recent_date = datetime.now(timezone.utc) - timedelta(days=1)
    for request_id, created_at in (("old", old_date), ("recent", recent_date)):
        store.record_request(
            request_id=request_id,
            method="GET",
            path="/health",
            status_code=200,
            duration_ms=1,
            request_payload=b"",
            response_payload=b"{}",
            created_at=created_at,
        )
        store.record_audit_event(
            "health_check",
            {"timestamp": created_at.isoformat(), "request_id": request_id},
        )

    assert store.purge_expired() == 2
    with engine.connect() as connection:
        request_ids = connection.execute(select(request_records.c.request_id)).scalars().all()
        audit_ids = connection.execute(select(audit_events.c.request_id)).scalars().all()
    assert request_ids == ["recent"]
    assert audit_ids == ["recent"]


def test_store_requires_valid_retention_and_encryption_key():
    with pytest.raises(ValueError, match="Fernet key"):
        DataStore("sqlite:///:memory:", "not-a-fernet-key")

    with pytest.raises(ValueError, match="at least 1"):
        DataStore("sqlite:///:memory:", Fernet.generate_key().decode(), retention_days=0)
