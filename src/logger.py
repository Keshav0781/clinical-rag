import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from google.cloud import bigquery
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# ─────────────────────────────────────────────
# Standard Python logger for internal messages
# Separate from the audit log
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# BigQuery configuration
# Dataset and table created in GCP Console
# Service account attached to Cloud Run
# handles authentication automatically
# ─────────────────────────────────────────────
PROJECT_ID = "project-df0cbdfe-9de3-4681-b3b"
DATASET_ID = "clinical_rag_logs"
TABLE_ID = "query_logs"
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# ─────────────────────────────────────────────
# Thread pool for async execution
# max_workers=2 — two background threads available
# for BigQuery writes without blocking the API
# ─────────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=2)

# ─────────────────────────────────────────────
# BigQuery client
# On Cloud Run this uses the attached service
# account automatically via Application Default
# Credentials — no credentials file needed
# ─────────────────────────────────────────────
client = bigquery.Client(project=PROJECT_ID)


# ─────────────────────────────────────────────
# Retry logic
# @retry decorator from tenacity library
# Retries up to 3 times on any Exception
# Waits: 1s → 2s → 4s between retries
# (exponential backoff — standard production pattern)
# Logs a warning before each retry attempt
# ─────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _write_to_bigquery(row: dict):
    """
    Internal function that writes one row to BigQuery.
    Decorated with @retry — automatically retries up to
    3 times with exponential backoff if write fails.

    Called by log_query_async in a background thread.
    Never called directly from api.py.

    Raises exception after all retries exhausted —
    caught by log_query_async which logs final warning.
    """
    errors = client.insert_rows_json(TABLE_REF, [row])
    if errors:
        raise Exception(f"BigQuery insert errors: {errors}")


def _execute_log(row: dict):
    """
    Wrapper that runs _write_to_bigquery with retry
    and handles final failure gracefully.

    Runs in a background thread via ThreadPoolExecutor.
    Any exception here is caught — API is never affected.
    """
    try:
        _write_to_bigquery(row)
        logger.info("Query logged to BigQuery successfully")
    except Exception as e:
        # All 3 retries exhausted — log final warning
        # In production this would also publish to
        # Cloud Pub/Sub dead letter queue
        logger.warning(f"Failed to log query after all retries: {e}")


def log_query(
    session_id: str,
    question: str,
    answer: str,
    sources: list,
    response_time: float,
    guardrail_triggered: bool,
    guardrail_reason: str,
    rerank_method: str,
    conversation_turn: int
):
    """
    Public function called by api.py after every query.

    ASYNC — submits BigQuery write to background thread
    and returns immediately. API response is NOT delayed
    by the logging operation.

    Interface is identical to previous synchronous version —
    api.py requires zero changes.

    Flow:
    1. Build row dictionary
    2. Submit to ThreadPoolExecutor (background thread)
    3. Return immediately — API sends response to user
    4. Background thread runs _execute_log with retry logic
    5. If all retries fail — warning logged, query entry lost
       (acceptable tradeoff vs blocking the API)
    """
    try:
        # Extract primary department from first source
        department = sources[0]["department"] if sources else "unknown"

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": json.dumps(sources),
            "response_time": response_time,
            "chunks_retrieved": len(sources),
            "guardrail_triggered": guardrail_triggered,
            "guardrail_reason": guardrail_reason,
            "rerank_method": rerank_method,
            "department": department,
            "conversation_turn": conversation_turn
        }

        # Submit to background thread — non-blocking
        # API returns response to user immediately
        # BigQuery write happens concurrently
        executor.submit(_execute_log, row)

    except Exception as e:
        # Row building failed — log warning, never crash API
        logger.warning(f"Failed to prepare log entry: {e}")


def get_stats():
    """
    Returns basic statistics from BigQuery audit log.
    Used by /stats endpoint for quick monitoring.
    Synchronous — called on demand, not on every request.
    """
    try:
        stats = {}

        # Total queries
        query = f"SELECT COUNT(*) as total FROM `{TABLE_REF}`"
        result = client.query(query).result()
        stats["total_queries"] = list(result)[0].total

        # Guardrail triggers
        query = f"""
            SELECT COUNT(*) as total
            FROM `{TABLE_REF}`
            WHERE guardrail_triggered = TRUE
        """
        result = client.query(query).result()
        stats["guardrail_triggers"] = list(result)[0].total

        # Average response time
        query = f"""
            SELECT ROUND(AVG(response_time), 3) as avg_time
            FROM `{TABLE_REF}`
            WHERE guardrail_triggered = FALSE
        """
        result = client.query(query).result()
        stats["avg_response_time"] = list(result)[0].avg_time

        # Queries by department
        query = f"""
            SELECT department, COUNT(*) as count
            FROM `{TABLE_REF}`
            WHERE department != 'unknown'
            GROUP BY department
            ORDER BY count DESC
        """
        result = client.query(query).result()
        stats["queries_by_department"] = {
            row.department: row.count for row in result
        }

        # Last 7 days
        query = f"""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM `{TABLE_REF}`
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        """
        result = client.query(query).result()
        stats["queries_last_7_days"] = {
            str(row.date): row.count for row in result
        }

        return stats

    except Exception as e:
        logger.warning(f"Failed to get stats from BigQuery: {e}")
        return {}