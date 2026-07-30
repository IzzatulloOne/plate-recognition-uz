"""SQLite-журнал распознанных номеров + снапшоты на диск."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("anpr.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    source      TEXT    NOT NULL,
    track_id    INTEGER,
    text        TEXT    NOT NULL,
    pretty      TEXT,
    conf        REAL,
    votes       INTEGER,
    format      TEXT,
    plate_class TEXT,
    color       TEXT,
    type_code   TEXT,
    box         TEXT,
    snapshot    TEXT,
    updated     INTEGER DEFAULT 0,   -- событие-уточнение (голосование передумало)
    previous    TEXT                 -- что было отдано до уточнения
);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_text ON events(text);
"""


class EventStore:
    #: как часто проверять размер журнала (в добавленных событиях)
    PRUNE_EVERY = 500

    def __init__(
        self,
        db_path: Path,
        snapshot_dir: Path,
        save_snapshots: bool = True,
        keep: int = 0,
    ):
        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.save_snapshots = save_snapshots
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._since_prune = 0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Догоняем схему в БД, созданных прошлыми версиями."""
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(events)")}
        for col, ddl in (("updated", "INTEGER DEFAULT 0"), ("previous", "TEXT")):
            if col not in have:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} {ddl}")

    # ------------------------------------------------------------------- запись
    def _write_snapshot(self, text: str, jpeg: bytes | None) -> str | None:
        if not (self.save_snapshots and jpeg):
            return None
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        folder = self.snapshot_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch for ch in text if ch.isalnum()) or "unknown"
        name = f"{int(time.time() * 1000)}_{safe}.jpg"
        path = folder / name
        path.write_bytes(jpeg)
        return str(path.relative_to(self.snapshot_dir))  # относительно snapshot_dir

    def add(self, event: dict, jpeg: bytes | None = None) -> int:
        snapshot = self._write_snapshot(event.get("text", ""), jpeg)
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO events
                   (ts, source, track_id, text, pretty, conf, votes, format,
                    plate_class, color, type_code, box, snapshot, updated, previous)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.get("ts", time.time()),
                    event.get("source", ""),
                    event.get("track_id"),
                    event.get("text", ""),
                    event.get("pretty", ""),
                    event.get("conf", 0.0),
                    event.get("votes", 0),
                    event.get("format"),
                    event.get("plate_class"),
                    event.get("color"),
                    event.get("type_code"),
                    str(event.get("box", [])),
                    snapshot,
                    int(bool(event.get("updated"))),
                    event.get("previous"),
                ),
            )
            self._conn.commit()
            event_id = int(cur.lastrowid or 0)
            self._since_prune += 1
        # журнал не должен расти бесконечно на долгоживущем сервере
        if self.keep and self._since_prune >= self.PRUNE_EVERY:
            self._since_prune = 0
            removed = self.prune(self.keep)
            if removed:
                log.info("журнал обрезан: удалено %d старых событий", removed)
        return event_id

    # ------------------------------------------------------------------- чтение
    def get(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        text: str | None = None,
        source: str | None = None,
        since: float | None = None,
        plate_class: str | None = None,
        color: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM events WHERE 1=1"
        args: list = []
        if text:
            sql += " AND text LIKE ?"
            args.append(f"%{text.upper()}%")
        if source:
            sql += " AND source = ?"
            args.append(source)
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        if plate_class:
            sql += " AND plate_class = ?"
            args.append(plate_class)
        if color:
            sql += " AND color = ?"
            args.append(color)
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
            by_class = self._conn.execute(
                "SELECT plate_class, COUNT(*) c FROM events GROUP BY plate_class"
            ).fetchall()
            by_color = self._conn.execute(
                "SELECT color, COUNT(*) c FROM events GROUP BY color"
            ).fetchall()
            last = self._conn.execute("SELECT MAX(ts) t FROM events").fetchone()["t"]
        return {
            "total": total,
            "by_class": {r["plate_class"] or "unknown": r["c"] for r in by_class},
            "by_color": {r["color"] or "unknown": r["c"] for r in by_color},
            "last_ts": last,
        }

    def prune(self, keep: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE id NOT IN "
                "(SELECT id FROM events ORDER BY ts DESC LIMIT ?)",
                (keep,),
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
