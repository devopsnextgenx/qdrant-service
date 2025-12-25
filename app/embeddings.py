import os
from typing import List, Optional
import httpx
import ollama
# from sentence_transformers import SentenceTransformer

class BaseEmbedder:
    dim: Optional[int] = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# class SentenceTransformersEmbedder(BaseEmbedder):
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.dim = self.model.get_sentence_embedding_dimension()

#     def encode(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, show_progress_bar=False)


class OllamaEmbedder(BaseEmbedder):
    """Use Ollama local HTTP endpoint to get embeddings.

    Configure with env vars:
      - EMBEDDING_BACKEND=ollama
      - OLLAMA_EMBED_MODEL (optional)
      - OLLAMA_EMBED_ENDPOINT (default: http://localhost:11434/api/embed)
    """

    def __init__(self, model_name: Optional[str] = None, endpoint: Optional[str] = None, timeout: int = 60):
        self.model = model_name or os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
        # Default endpoint includes the /api/embed path; allow users to set either the host or the full embed path.
        endpoint = endpoint or os.getenv("OLLAMA_EMBED_ENDPOINT", "http://localhost:11434")
        # Normalize to always end with /embed (covers both /embed and /api/embed)
        if not endpoint.rstrip('/').endswith('/embed'):
            endpoint = endpoint.rstrip('/') + '/embed'
        self.endpoint = endpoint
        self.timeout = timeout
        # Initialize ollama client if the package is installed. The client expects a host like "http://localhost:11434" (without the /embed path).
        if ollama is not None:
            try:
                host = self.endpoint
                if host.endswith("/embed"):
                    host = host.rsplit("/embed", 1)[0]
                self.ollama_client = ollama.Client(host=host)
            except Exception as e:
                print(f"Failed to initialize ollama.Client: {e!r}")
                import traceback
                traceback.print_exc()
                self.ollama_client = None
        else:
            self.ollama_client = None
    
    
    def _extract_embeddings(self, resp):
        # Ollama official client response
        if hasattr(resp, "embeddings"):
            return resp.embeddings

        # HTTP fallback (dict)
        if isinstance(resp, dict):
            return resp["embeddings"]

        raise TypeError(f"Unsupported Ollama response type: {type(resp)}")


    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        data = None

        # Prefer using the official ollama client when available; fall back to direct HTTP otherwise
        # Try common client method signature
        resp = None
        try:
            resp = self.ollama_client.embed(model=self.model, input=texts)
        except Exception as e:
            print(f"Failed to use ollama client Exception: {e!r}")

        embeddings = self._extract_embeddings(resp)

        if embeddings and self.dim is None:
            self.dim = len(embeddings[0])
            print(f"Ollama embeddings dim: {self.dim}")
        return embeddings if embeddings else []

    def health(self) -> dict:
        """Basic health probe for the Ollama embedding endpoint"""
        try:
            # send a tiny request and ensure we get a JSON response
            payload = {"model": self.model, "input": ["__health_check__"]}
            if getattr(self, "ollama_client", None) is not None:
                try:
                    resp = self.ollama_client.embed(model=self.model, input=["__health_check__"], timeout=5)
                except TypeError:
                    # Some client versions accept a single payload dict; pass it as kwargs to avoid
                    # the dict being treated as the 'model' positional argument.
                    resp = self.ollama_client.embed(**payload, timeout=5)
                data = resp.json() if hasattr(resp, "json") else resp
            else:
                resp = httpx.post(self.endpoint, json=payload, timeout=5)
                resp.raise_for_status()
                data = resp.json()

            # best-effort model info
            return {"ok": True, "model": self.model, "sample": (data if isinstance(data, dict) else None)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# Factory to pick backend using project config
from .config import CONFIG

BACKEND = CONFIG.get("embeddings", {}).get("backend", "ollama").lower()
if BACKEND == "ollama":
    model = CONFIG.get("embeddings", {}).get("ollama", {}).get("model")
    endpoint = CONFIG.get("embeddings", {}).get("ollama", {}).get("endpoint")
    timeout = CONFIG.get("embeddings", {}).get("ollama", {}).get("timeout", 60)
    print(f"Using Ollama embeddings backend with model {model} at {endpoint}")
    embedder = OllamaEmbedder(model_name=model, endpoint=endpoint, timeout=timeout)
else:
    # model = CONFIG.get("embeddings", {}).get("sentence_transformers", {}).get("model", "all-MiniLM-L6-v2")
    # embedder = SentenceTransformersEmbedder(model_name=model)
    # logger.warning(f"Embeddings backend {BACKEND} not implemented, using Ollama instead")
    print(f"Embeddings backend {BACKEND} not implemented, using Ollama instead")
    model = CONFIG.get("embeddings", {}).get("ollama", {}).get("model")
    endpoint = CONFIG.get("embeddings", {}).get("ollama", {}).get("endpoint")
    timeout = CONFIG.get("embeddings", {}).get("ollama", {}).get("timeout", 60)
    print(f"Using Ollama embeddings backend with model {model} at {endpoint}")
    embedder = OllamaEmbedder(model_name=model, endpoint=endpoint, timeout=timeout)