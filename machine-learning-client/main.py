"""Application entry point for the machine learning client API."""

from src.server import app


def main() -> None:
    """Run the ML prediction API service."""
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
