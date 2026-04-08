"""Test cases for the Flask web application."""

import pytest
from app import app


@pytest.fixture(name="client")
def fixture_client():
    """Test client for the app."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_route(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "service": "web-app"}
