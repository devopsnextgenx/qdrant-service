import os
import sys
import pytest

# Ensure project root is on sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embeddings import OllamaEmbedder


def test_ollama_client_embed_with_kwargs():
    embedder = OllamaEmbedder(model_name="my-model", endpoint="http://localhost:11434")

    class MockResp:
        def json(self):
            return {"embeddings": [[1.0, 2.0]]}

    class MockClient:
        def embed(self, *args, **kwargs):
            # If called with kwargs, return a mock response
            if kwargs:
                return MockResp()
            # If called positionally with a dict, simulate a validation error
            if args and isinstance(args[0], dict):
                raise ValueError("validation error: model must be string")
            raise TypeError("unexpected call signature")

    embedder.ollama_client = MockClient()
    vecs = embedder.encode(["hello world"])
    assert vecs == [[1.0, 2.0]]


def test_ollama_client_embed_fallback_to_http(monkeypatch):
    # Simulate client unavailable and HTTP POST returning data
    embedder = OllamaEmbedder(model_name="my-model", endpoint="http://localhost:11434")

    class MockResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_post(url, json, timeout):
        assert url.endswith("/embed")
        assert json["model"] == "my-model"
        return MockResponse({"embeddings": [[0.5, 0.6]]})

    # Disable client to force HTTP path
    embedder.ollama_client = None
    monkeypatch.setattr("httpx.post", fake_post)

    vecs = embedder.encode(["hi"])
    assert vecs == [[0.5, 0.6]]

    vecs = embedder.encode(["message one", "message two", "message three"])    assert isinstance(vecs, list)


def test_ollama_response_with_metadata_and_embeds(monkeypatch):
    # Simulate a metadata-heavy response where "embeddings" is a dict mapping indices to vectors
    embedder = OllamaEmbedder(model_name="my-model", endpoint="http://localhost:11434")

    class MockResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    fake = {
        "model": "embeddinggemma:latest",
        "created_at": None,
        "embeddings": {
            "0": [0.1, 0.2],
            "1": [0.3, 0.4]
        }
    }

    def fake_post(url, json, timeout):
        assert url.endswith("/embed")
        return MockResponse(fake)

    embedder.ollama_client = None
    monkeypatch.setattr("httpx.post", fake_post)

    vecs = embedder.encode(["a", "b"])
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]


def test_ollama_nested_data_structures(monkeypatch):
    # Response with embeddings nested inside data->list of dicts
    embedder = OllamaEmbedder(model_name="my-model", endpoint="http://localhost:11434")

    class MockResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    fake = {
        "meta": {"info": "x"},
        "something": {
            "data": [
                {"embedding": [0.7, 0.8]},
                {"embedding": [0.9, 1.0]}
            ]
        }
    }

    def fake_post(url, json, timeout):
        return MockResponse(fake)

    embedder.ollama_client = None
    monkeypatch.setattr("httpx.post", fake_post)

    vecs = embedder.encode(["one", "two"])
    assert vecs == [[0.7, 0.8], [0.9, 1.0]]    