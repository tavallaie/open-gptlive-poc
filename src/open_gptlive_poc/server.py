"""ASGI entry point for local development."""

from .app import create_app
from .config import Settings


app = create_app(Settings.from_env())
