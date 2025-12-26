import os
import yaml
from typing import Set, Dict, List

class QdrantProcessTracker:
    def __init__(self, tracker_path: str = "config/qdrant-process-tracker.yml"):
        self.tracker_path = tracker_path
        self.processed_files: Dict[str, List[str]] = {
            "captions": [],
            "stories": []
        }
        self.load()

    def load(self):
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        self.processed_files["captions"] = data.get("captions", [])
                        self.processed_files["stories"] = data.get("stories", [])
            except Exception as e:
                print(f"Error loading tracker file: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)
        try:
            with open(self.tracker_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.processed_files, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving tracker file: {e}")

    def is_processed(self, file_path: str, category: str) -> bool:
        return file_path in self.processed_files.get(category, [])

    def mark_as_processed(self, file_path: str, category: str):
        if category not in self.processed_files:
            self.processed_files[category] = []
        if file_path not in self.processed_files[category]:
            self.processed_files[category].append(file_path)

    def get_processed_files(self, category: str) -> Set[str]:
        return set(self.processed_files.get(category, []))
