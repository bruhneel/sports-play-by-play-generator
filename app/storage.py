"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def prepare(destination: Path = Path("artifacts/corpus.json"), seed: int = 42) -> dict:
    """Build reproducible train and validation records from local data."""
    from app.core import clean_lines, split_lines

    train, val = split_lines(clean_lines(Path("data/commentary.txt").read_text(encoding="utf-8")), seed)
    news = clean_lines(Path("data/news.txt").read_text(encoding="utf-8"))
    result = {"train": train + news, "val": val, "news_lines": len(news), "seed": seed}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
