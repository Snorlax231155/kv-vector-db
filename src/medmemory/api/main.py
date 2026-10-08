"""ASGI entry point for uvicorn: `uvicorn medmemory.api.main:app`."""

from medmemory.api.app import create_app

app = create_app()
