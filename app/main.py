import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.auth import bootstrap_admin, cookie_secure
from app.data_store import get_data_store
from app.observability import AUDIT_EVENTS, configure_observability, log_event
from app.prediction import MODEL_PATH, _load_model
from app.routes import router
from app.schemas import Features, PredictionRequest, PredictionResponse


async def _retention_worker():
    store = get_data_store()
    while True:
        await asyncio.sleep(24 * 60 * 60)
        if store is not None:
            await asyncio.to_thread(store.purge_expired)


@asynccontextmanager
async def lifespan(_: FastAPI):
    cookie_secure()
    store = get_data_store()
    retention_task = None
    if store is not None:
        await asyncio.to_thread(store.initialize)
        await asyncio.to_thread(bootstrap_admin, store)
        await asyncio.to_thread(store.purge_expired)
        retention_task = asyncio.create_task(_retention_worker())
    try:
        yield
    finally:
        if retention_task is not None:
            retention_task.cancel()
            try:
                await retention_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Readmission Risk API", version="1.0.0", lifespan=lifespan)
configure_observability(app)
app.include_router(router)

__all__ = [
    "AUDIT_EVENTS",
    "Features",
    "MODEL_PATH",
    "PredictionRequest",
    "PredictionResponse",
    "_load_model",
    "app",
    "log_event",
]
