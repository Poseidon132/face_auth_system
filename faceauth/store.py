"""Local SQLite embedding registry. No photographs are persisted.

Embeddings remain sensitive biometric data. Database is local demo storage,
not encrypted, not suitable as a production authentication authority.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_DB, MAX_ENROLLMENTS_PER_PERSON
from .vision import normalize_feature


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_person_name(name: str) -> str:
    if not isinstance(name, str) or any(ord(c) < 32 for c in name):
        raise ValueError("姓名或代号不能包含控制字符。")
    value = " ".join(name.strip().split())
    if not 1 <= len(value) <= 50:
        raise ValueError("姓名或代号长度必须在 1 到 50 个字符之间。")
    if any(ord(c) < 32 for c in value):
        raise ValueError("姓名或代号不能包含控制字符。")
    return value


class FaceStore:
    def __init__(self, database: str | Path = DEFAULT_DB, *, model_id: str) -> None:
        if not model_id.strip():
            raise ValueError("model_id 不能为空")
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_key TEXT UNIQUE NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS templates (
                    template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    feature_size INTEGER NOT NULL,
                    detection_score REAL NOT NULL,
                    created_utc TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_templates_person ON templates(person_id);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    subject TEXT,
                    accepted INTEGER NOT NULL,
                    similarity REAL,
                    threshold REAL,
                    note TEXT NOT NULL
                );
            """)
            old = conn.execute("SELECT value FROM settings WHERE key='model_id'").fetchone()
            if old and old["value"] != model_id:
                raise RuntimeError(
                    "数据库由另一个特征提取模型创建。请备份数据库并重新注册，不能混合两种模型的向量。"
                )
            if not old:
                conn.execute("INSERT INTO settings(key,value) VALUES('model_id', ?)", (model_id,))

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=10)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA busy_timeout = 10000")
            con.execute("PRAGMA secure_delete = ON")
            with con:
                yield con
        finally:
            con.close()

    def persons(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT p.person_id,p.name,p.created_utc,COUNT(t.template_id) AS samples
                FROM persons p LEFT JOIN templates t ON t.person_id=p.person_id
                GROUP BY p.person_id ORDER BY p.name_key
            """).fetchall()
            return [dict(row) for row in rows]

    def person_by_name(self, name: str) -> dict[str, Any] | None:
        key = validate_person_name(name).casefold()
        with self._connect() as con:
            row = con.execute("SELECT * FROM persons WHERE name_key=?", (key,)).fetchone()
            return dict(row) if row else None

    def sample_count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM templates").fetchone()[0])

    def enroll(self, name: str, embedding: np.ndarray, *, detection_score: float, consent: bool) -> dict[str, Any]:
        if not consent:
            raise PermissionError("必须确认已获得本人同意才能注册其人脸特征。")
        name = validate_person_name(name)
        if not np.isfinite(detection_score) or not 0.0 <= detection_score <= 1.0:
            raise ValueError("检测分数无效")
        feature = normalize_feature(embedding)
        key = name.casefold()
        now = utc_now()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT person_id,name FROM persons WHERE name_key=?", (key,)).fetchone()
            if existing:
                pid = existing["person_id"]
                display_name = existing["name"]
            else:
                pid = uuid.uuid4().hex
                display_name = name
                con.execute(
                    "INSERT INTO persons(person_id,name,name_key,created_utc) VALUES(?,?,?,?)",
                    (pid, display_name, key, now),
                )
            count = int(con.execute("SELECT COUNT(*) FROM templates WHERE person_id=?", (pid,)).fetchone()[0])
            if count >= MAX_ENROLLMENTS_PER_PERSON:
                raise ValueError(f"每个人最多注册 {MAX_ENROLLMENTS_PER_PERSON} 张样本，请先删除旧身份并重新注册。")
            prev_size = con.execute("SELECT feature_size FROM templates LIMIT 1").fetchone()
            if prev_size and prev_size[0] != feature.size:
                raise ValueError("特征维度不一致，检查模型与数据库是否匹配。")
            con.execute(
                "INSERT INTO templates(person_id,embedding,feature_size,detection_score,created_utc) VALUES(?,?,?,?,?)",
                (pid, sqlite3.Binary(feature.tobytes()), len(feature), float(detection_score), now),
            )
            return {"person_id": pid, "name": display_name, "samples": count+1}

    def templates_for(self, person_id: str | None = None) -> list[tuple[str, str, np.ndarray]]:
        with self._connect() as con:
            if person_id is None:
                rows = con.execute("""
                    SELECT p.person_id,p.name,t.embedding,t.feature_size
                    FROM templates t JOIN persons p ON p.person_id=t.person_id
                    ORDER BY p.name,t.template_id
                """).fetchall()
            else:
                rows = con.execute("""
                    SELECT p.person_id,p.name,t.embedding,t.feature_size
                    FROM templates t JOIN persons p ON p.person_id=t.person_id
                    WHERE p.person_id=? ORDER BY t.template_id
                """, (person_id,)).fetchall()
        result = []
        for row in rows:
            if len(row["embedding"]) % 4 != 0:
                raise RuntimeError("数据库特征向量长度不合法，可能存在损坏。")
            vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if len(vec) != row["feature_size"]:
                raise RuntimeError("数据库特征向量长度不合法，可能存在损坏。")
            result.append((row["person_id"], row["name"], vec))
        return result

    def delete_person(self, person_id: str) -> bool:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT name FROM persons WHERE person_id=?", (person_id,)).fetchone()
            if not row:
                return False
            cur = con.execute("DELETE FROM persons WHERE person_id=?", (person_id,))
            # Remove name-bearing audit events as part of the data erasure action.
            con.execute("DELETE FROM events WHERE subject=?", (row["name"],))
            return cur.rowcount == 1

    def record_event(
        self, operation: str, *, subject: str | None, accepted: bool,
        similarity: float | None, threshold: float | None, note: str = ""
    ) -> None:
        if operation not in {"verify", "identify"}:
            raise ValueError("未知事件类型")
        with self._connect() as con:
            con.execute(
                "INSERT INTO events(created_utc,operation,subject,accepted,similarity,threshold,note)"
                " VALUES(?,?,?,?,?,?,?)",
                (utc_now(), operation, subject, int(bool(accepted)), similarity, threshold, note[:100]),
            )

    def logs(self, limit: int = 250) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        with self._connect() as con:
            rows = con.execute(
                "SELECT created_utc,operation,subject,accepted,similarity,threshold,note "
                "FROM events ORDER BY event_id DESC LIMIT ?", (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def export_logs_csv(self, limit: int = 5000) -> bytes:
        result = io.StringIO()
        fields = ["created_utc", "operation", "subject", "accepted", "similarity", "threshold", "note"]
        writer = csv.DictWriter(result, fieldnames=fields)
        writer.writeheader()
        writer.writerows(self.logs(limit))
        return result.getvalue().encode("utf-8-sig")

    def clear_logs(self) -> int:
        with self._connect() as con:
            cur = con.execute("DELETE FROM events")
            return cur.rowcount
