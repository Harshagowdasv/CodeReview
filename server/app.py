"""
server/app.py — OpenEnv entry point.
The openenv validator requires this file as the server entry point.
It simply re-exports the FastAPI app from main.py and provides a main() launcher.
"""

import uvicorn
from server.main import app  # noqa: F401  re-export for openenv discovery


def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()