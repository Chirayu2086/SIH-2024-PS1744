import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path("./outputs/audit_log.jsonl")


def log_event(event_type: str, payload: dict) -> None:
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "payload": payload,
    }

    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
