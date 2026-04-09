"""Main web application for Ergonomics Monitor."""

from flask import Flask

# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/")
def index():
    """Return health check response."""
    return {"status": "ok", "service": "web-app"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
