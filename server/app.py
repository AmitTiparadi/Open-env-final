"""Compatibility wrapper expected by the OpenEnv CLI template."""

from incident_commander_env.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
