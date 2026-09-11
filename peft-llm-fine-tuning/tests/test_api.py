from fastapi.testclient import TestClient

import peft_summarizer.api as api


def test_health_without_model(monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
