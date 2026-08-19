"""Docker SDK entrypoint: re-export the FastAPI service so `uvicorn app:app` works."""
from service import app  # noqa: F401
