# -*- coding: utf-8 -*-
"""HTTP API for the 104 lead scoring and outreach workflow.

Typical n8n flow:
1. POST /weekly-leads to start a background crawl + scoring job.
2. Poll GET /jobs/<job_id> until status is "done".
3. Fetch GET /jobs/<job_id>/top-leads and send emails in n8n.
4. POST /jobs/<job_id>/mark-developed after successful email dispatch.
"""

from __future__ import annotations

import traceback
import uuid
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import pandas as pd
from flask import Flask, Response, request, send_file

from cosine_similarity_analysis import (
    DEFAULT_CRAWL_TARGET,
    DEFAULT_CUSTOMER_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POTENTIAL_FILE,
    DEFAULT_RECENT_DAYS,
    DEFAULT_MIN_CRAWL_SUCCESS_RATIO,
    DEFAULT_TOP_N,
    append_development_history,
    load_development_history,
    now_iso,
    run_weekly_pipeline,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5001
JOBS_DIR = PROJECT_DIR / "outputs" / "api_jobs"
STALE_JOB_AFTER = timedelta(hours=3)

app = Flask(__name__)
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()

COMPACT_TOP_LEAD_COLUMNS = [
    "outreach_rank",
    "rank",
    "企業名稱",
    "Email",
    "email",
    "contact_email",
    "電子郵件",
    "email_status",
    "email_source_url",
    "email_candidates",
    "similarity_score",
    "score_percentile",
    "manual_physical_letter",
    "n8n_should_email",
    "104_custNo",
    "104_official_company_name",
    "104_website",
    "福利制度",
    "104_welfare_snack_keywords",
    "104_address",
    "104_address_city",
    "104_address_district",
    "local_employee_count",
    "local_capital_ntd",
    "industry",
    "top_existing_customer_names",
    "top_existing_customer_scores",
    "last_developed_at",
    "recently_developed",
]


def resolve_path(value: Any, default: str | Path) -> Path:
    path = Path(str(value or default))
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def bool_from_payload(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def truthy_query(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        sanitize_json_value(record)
        for record in df.to_dict(orient="records")
    ]


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if hasattr(value, "item"):
        try:
            return sanitize_json_value(value.item())
        except (TypeError, ValueError):
            pass

    return value


def json_response(payload: Any, status: int = 200) -> Response:
    body = json.dumps(
        sanitize_json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
    )
    return Response(body, status=status, mimetype="application/json")


def build_pipeline_config(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = resolve_path(payload.get("output_dir"), DEFAULT_OUTPUT_DIR)
    history_file = resolve_path(
        payload.get("history_file"),
        output_dir / "development_history.csv",
    )
    top_leads_file = resolve_path(
        payload.get("top_leads_file"),
        output_dir / "n8n_top100_leads.csv",
    )

    if "crawl_before_score" in payload:
        crawl_before_score = bool_from_payload(payload, "crawl_before_score", True)
    elif "crawl" in payload:
        crawl_before_score = bool_from_payload(payload, "crawl", True)
    else:
        crawl_before_score = True

    return {
        "customers_path": resolve_path(payload.get("customers"), DEFAULT_CUSTOMER_FILE),
        "potentials_path": resolve_path(payload.get("potentials"), DEFAULT_POTENTIAL_FILE),
        "output_dir": output_dir,
        "history_file": history_file,
        "crawl_before_score": crawl_before_score,
        "target_leads": int(payload.get("target_leads", DEFAULT_CRAWL_TARGET)),
        "max_candidates": int_or_none(payload.get("max_candidates")),
        "recent_days": int(payload.get("recent_days", DEFAULT_RECENT_DAYS)),
        "top_n": int(payload.get("top_n", DEFAULT_TOP_N)),
        "top_leads_file": top_leads_file,
        "mark_top_n_developed": bool_from_payload(payload, "mark_top_n_developed", False),
        "enrich_emails": bool_from_payload(payload, "enrich_emails", False),
        "min_crawl_success_ratio": float(
            payload.get("min_crawl_success_ratio", DEFAULT_MIN_CRAWL_SUCCESS_RATIO)
        ),
    }


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in config.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def job_state_path(job_id: str) -> Path:
    safe_job_id = "".join(ch for ch in job_id if ch.isalnum() or ch in "-_")
    return JOBS_DIR / f"{safe_job_id}.json"


def persist_job(job_id: str, job: dict[str, Any]) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_state_path(job_id).write_text(
        json.dumps(sanitize_json_value(job), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_persisted_job(job_id: str) -> dict[str, Any] | None:
    path = job_state_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def fail_stale_job_if_needed(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") not in {"queued", "running"}:
        return dict(job)

    started_at = parse_iso_datetime(job.get("started_at") or job.get("created_at"))
    if not started_at:
        return dict(job)

    now = datetime.now(started_at.tzinfo) if started_at.tzinfo else datetime.now()
    if now - started_at < STALE_JOB_AFTER:
        return dict(job)

    stale_job = dict(job)
    stale_job.update(
        {
            "status": "failed",
            "finished_at": now_iso(),
            "error": (
                "job_stale: job stayed queued/running longer than "
                f"{STALE_JOB_AFTER.total_seconds() / 3600:.0f} hours"
            ),
        }
    )
    with jobs_lock:
        jobs[job_id] = stale_job
        persist_job(job_id, stale_job)
    return dict(stale_job)


def set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)
        persist_job(job_id, jobs[job_id])


def get_job(job_id: str) -> dict[str, Any] | None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job_copy = dict(job)
        else:
            job_copy = None

    if job_copy:
        return fail_stale_job_if_needed(job_id, job_copy)

    persisted = load_persisted_job(job_id)
    if persisted:
        with jobs_lock:
            jobs[job_id] = persisted
        return fail_stale_job_if_needed(job_id, persisted)

    return None


def run_job(job_id: str, config: dict[str, Any]) -> None:
    set_job(job_id, status="running", started_at=now_iso())
    try:
        summary = run_weekly_pipeline(run_id=job_id, **config)
        set_job(job_id, status="done", finished_at=now_iso(), result=summary)
    except Exception as exc:
        set_job(
            job_id,
            status="failed",
            finished_at=now_iso(),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


@app.get("/health")
def health() -> Any:
    return json_response({"ok": True, "service": "104-lead-scoring-api", "time": now_iso()})


@app.post("/weekly-leads")
def weekly_leads() -> Any:
    payload = request.get_json(silent=True) or {}
    config = build_pipeline_config(payload)
    async_mode = bool_from_payload(payload, "async", True)
    job_id = str(payload.get("job_id") or uuid.uuid4())

    set_job(
        job_id,
        status="queued",
        created_at=now_iso(),
        config=public_config(config),
    )

    if async_mode:
        thread = Thread(target=run_job, args=(job_id, config), daemon=True)
        thread.start()
        return json_response(
            {
                "job_id": job_id,
                "status": "queued",
                "status_url": f"/jobs/{job_id}",
                "top_leads_url": f"/jobs/{job_id}/top-leads",
            }
        ), 202

    run_job(job_id, config)
    return json_response(get_job(job_id))


@app.get("/jobs/<job_id>")
def job_status(job_id: str) -> Any:
    job = get_job(job_id)
    if not job:
        return json_response({"error": "job_not_found", "job_id": job_id}, status=404)
    return json_response(job)


@app.get("/jobs/<job_id>/top-leads")
def job_top_leads(job_id: str) -> Any:
    job = get_job(job_id)
    if not job:
        return json_response({"error": "job_not_found", "job_id": job_id}, status=404)
    if job.get("status") != "done":
        return json_response({"error": "job_not_done", "job_id": job_id, "status": job.get("status")}, status=409)

    top_leads_csv = Path(job["result"]["top_leads_csv"])
    if not top_leads_csv.exists():
        return json_response({"error": "top_leads_file_not_found", "path": str(top_leads_csv)}, status=404)

    limit = int(request.args.get("limit", 0) or 0)
    df = pd.read_csv(top_leads_csv, encoding="utf-8-sig")
    if limit > 0:
        df = df.head(limit)
    if truthy_query(request.args.get("compact"), default=False):
        keep_columns = [column for column in COMPACT_TOP_LEAD_COLUMNS if column in df.columns]
        df = df[keep_columns].copy()

    return json_response(
        {
            "job_id": job_id,
            "count": int(len(df)),
            "leads": dataframe_to_records(df),
        }
    )


@app.get("/jobs/<job_id>/files/<file_key>")
def job_file(job_id: str, file_key: str) -> Any:
    job = get_job(job_id)
    if not job:
        return json_response({"error": "job_not_found", "job_id": job_id}, status=404)
    if job.get("status") != "done":
        return json_response({"error": "job_not_done", "job_id": job_id, "status": job.get("status")}, status=409)

    allowed = {
        "ranking_csv",
        "ranking_xlsx",
        "ranking_snapshot_csv",
        "ranking_snapshot_xlsx",
        "top_matches_csv",
        "top_leads_csv",
        "top_leads_xlsx",
    }
    if file_key not in allowed:
        return json_response({"error": "unsupported_file_key", "allowed": sorted(allowed)}, status=400)

    path = Path(job["result"][file_key])
    if not path.exists():
        return json_response({"error": "file_not_found", "path": str(path)}, status=404)
    return send_file(path, as_attachment=True)


@app.post("/jobs/<job_id>/mark-developed")
def mark_job_developed(job_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    job = get_job(job_id)
    if not job:
        return json_response({"error": "job_not_found", "job_id": job_id}, status=404)
    if job.get("status") != "done":
        return json_response({"error": "job_not_done", "job_id": job_id, "status": job.get("status")}, status=409)

    top_leads_csv = Path(job["result"]["top_leads_csv"])
    history_file = Path(job["result"]["history_file"])
    top_n = int(payload.get("top_n", job["result"]["top_leads_count"]))
    status = str(payload.get("status", "developed"))
    source = str(payload.get("source", "n8n"))
    outcome = payload.get("outcome")
    outcome_at = payload.get("outcome_at")

    leads = pd.read_csv(top_leads_csv, encoding="utf-8-sig").head(top_n)
    history = append_development_history(
        leads,
        history_file,
        source=source,
        status=status,
        outcome=str(outcome) if outcome is not None else None,
        outcome_at=str(outcome_at) if outcome_at is not None else None,
        run_id=job_id,
    )
    return json_response(
        {
            "job_id": job_id,
            "marked_count": int(len(leads)),
            "history_file": str(history_file),
            "history_rows": int(len(history)),
        }
    )


@app.post("/development-history")
def append_history() -> Any:
    payload = request.get_json(silent=True) or {}
    if isinstance(payload, list):
        history_file = resolve_path(None, Path(DEFAULT_OUTPUT_DIR) / "development_history.csv")
        leads_payload = payload
        source = "n8n"
        status = "developed"
        outcome = None
        outcome_at = None
        run_id = str(uuid.uuid4())
    else:
        history_file = resolve_path(
            payload.get("history_file"),
            Path(DEFAULT_OUTPUT_DIR) / "development_history.csv",
        )
        leads_payload = payload.get("leads", [])
        source = str(payload.get("source", "n8n"))
        status = str(payload.get("status", "developed"))
        outcome = payload.get("outcome")
        outcome_at = payload.get("outcome_at")
        run_id = str(payload.get("run_id") or uuid.uuid4())

    if not isinstance(leads_payload, list) or not leads_payload:
        return json_response({"error": "payload must include non-empty leads list"}, status=400)

    leads = pd.DataFrame(leads_payload)
    history = append_development_history(
        leads,
        history_file,
        source=source,
        status=status,
        outcome=str(outcome) if outcome is not None else None,
        outcome_at=str(outcome_at) if outcome_at is not None else None,
        run_id=run_id,
    )
    return json_response(
        {
            "marked_count": int(len(leads)),
            "history_file": str(history_file),
            "history_rows": int(len(history)),
        }
    )


@app.get("/development-history")
def get_history() -> Any:
    history_file = resolve_path(
        request.args.get("history_file"),
        Path(DEFAULT_OUTPUT_DIR) / "development_history.csv",
    )
    history = load_development_history(history_file)
    limit = int(request.args.get("limit", 100) or 100)
    return json_response(
        {
            "history_file": str(history_file),
            "count": int(len(history)),
            "rows": dataframe_to_records(history.tail(limit)),
        }
    )


if __name__ == "__main__":
    print(f"API Server starting at http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT)
