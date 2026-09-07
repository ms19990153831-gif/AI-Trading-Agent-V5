"""High-level memory: recent experiences used as context by agents."""

from __future__ import annotations


class Memory:
    def __init__(self, db) -> None:
        self.db = db

    def search(self, limit: int = 5) -> list[dict]:
        return self.db.get_recent_experiences(limit)

    def save_experience(self, content: str, category: str = "lesson") -> int:
        return self.db.save_experience(content, category)

    def context_block(self, limit: int = 5) -> list[dict]:
        return [
            {"content": row["content"], "category": row["category"]}
            for row in self.search(limit)
        ]
