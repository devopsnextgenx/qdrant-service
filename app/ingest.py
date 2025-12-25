import os
import yaml
import uuid
from typing import Iterable, Dict, Any, List

DATA_DIR = os.getenv("DATA_DIR", "data")


def _iter_yaml_files(path: str):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".yml") or f.endswith(".yaml"):
                yield os.path.join(root, f)


def extract_text_fields_from_dict(d: Dict[str, Any]) -> List[str]:
    """Recursively collect string values from a nested dict/list"""
    out = []
    if isinstance(d, dict):
        for v in d.values():
            out.extend(extract_text_fields_from_dict(v))
    elif isinstance(d, list):
        for v in d:
            out.extend(extract_text_fields_from_dict(v))
    elif isinstance(d, str):
        out.append(d.strip())
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def iter_captions(data_dir: str = DATA_DIR) -> Iterable[Dict[str, Any]]:
    captions_dir = os.path.join(data_dir, "captions")
    if not os.path.exists(captions_dir):
        return
    for file in _iter_yaml_files(captions_dir):
        doc = load_yaml(file)
        # try to extract meaningful fields
        corrected = (doc.get("text_processing") or {}).get("corrected_text")
        translated = (doc.get("translation") or {}).get("translated_text") or (doc.get("text_processing") or {}).get("translated_text")
        full_text = (doc.get("ocr") or {}).get("full_text") or doc.get("full_text")

        # fallbacks
        texts = list(filter(None, [corrected, translated, full_text]))
        if not texts:
            texts = extract_text_fields_from_dict(doc)

        for t in texts:
            yield {
                "id": str(uuid.uuid4()),
                "text": t,
                "payload": {
                    "source": file,
                    "type": "caption",
                },
            }


def iter_stories(data_dir: str = DATA_DIR) -> Iterable[Dict[str, Any]]:
    stories_dir = os.path.join(data_dir, "stories")
    if not os.path.exists(stories_dir):
        return
    # Each thread has ymls folder with page_*.yml
    for root, dirs, files in os.walk(stories_dir):
        for f in files:
            if not f.endswith(".yml") and not f.endswith(".yaml"):
                continue
            path = os.path.join(root, f)
            doc = load_yaml(path)
            # if there's an explicit posts list
            if isinstance(doc, dict) and "posts" in doc and isinstance(doc["posts"], list):
                for i, post in enumerate(doc["posts"]):
                    if isinstance(post, str):
                        t = post
                    else:
                        # collect text-like fields
                        t = post.get("content")
                        # all_texts = extract_text_fields_from_dict(post)
                        # t = "\n".join(all_texts)
                    if t.strip():
                        yield {"id": str(uuid.uuid4()), "text": t, "payload": {"source": path, "type": "story", "page": f, "post_index": i}}
            else:
                # fallback: treat page as single document containing all string fields
                texts = extract_text_fields_from_dict(doc)
                if texts:
                    yield {"id": str(uuid.uuid4()), "text": "\n".join(texts), "payload": {"source": path, "type": "story", "page": f}}


def iter_all_documents(data_dir: str = DATA_DIR) -> Iterable[Dict[str, Any]]:
    for d in iter_captions(data_dir=data_dir):
        yield d
    for d in iter_stories(data_dir=data_dir):
        yield d
