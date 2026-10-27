"""Clients for the decision-model APIs, with an on-disk response cache.

Each record is one call per model with the identical question list. Only the returned
probabilities are cached (never the record itself), so the cache can be published.
"""

import datetime as dt
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("data/api_responses")


class DecisionAPI:
    name: str
    url: str
    model: str
    key_env: str

    def headers(self):
        return {
            "Authorization": f"Bearer {os.environ[self.key_env]}",
            "Content-Type": "application/json",
        }

    def available(self):
        return bool(os.environ.get(self.key_env))


class Grayson(DecisionAPI):
    # https://docs.finic.ai/reference/decide
    name, url, model, key_env = "grayson", "https://api.finic.ai/v1/decide", "grayson-1", "GRAYSON_API_KEY"

    def body(self, context, questions):
        qs = {k: {"type": "noul", "instructions": q} for k, q in questions.items()}
        return {"model": self.model, "context": context, "questions": qs}

    def parse(self, resp):
        return {k: a["probability"] for k, a in resp["answers"].items()}


class Jev(DecisionAPI):
    # https://docs.typesafe.ai/api/system-one
    name, url, model, key_env = "jev", "https://api.typesafe.ai/v1/systemone", "jev-1.14.0", "TYPESAFE_API_KEY"

    def body(self, context, questions):
        qs = {k: {"type": "noul", "instructions": q} for k, q in questions.items()}
        return {"model": self.model, "state": context, "questions": qs}

    def parse(self, resp):
        return {k: a["noul"] for k, a in resp["answers"].items()}


RETRY_STATUS = {429, 500, 502, 503, 504}


class MissingKey(RuntimeError):
    """Responses are missing from the cache and no API key is set to fetch them."""


class Throttle:
    """Spaces calls across threads to stay under a requests-per-minute limit."""

    def __init__(self, per_minute):
        self.interval = 60 / per_minute
        self.next = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.interval
            delay = self.next - self.interval - now
        if delay > 0:
            time.sleep(delay)


def _call(api, context, questions, session, throttle, max_tries=8):
    status = None
    for attempt in range(max_tries):
        throttle.wait()
        try:
            r = session.post(api.url, headers=api.headers(), json=api.body(context, questions), timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            r = None
        if r is not None and r.status_code == 200:
            return r.json()
        if r is not None and r.status_code not in RETRY_STATUS:
            raise RuntimeError(f"{api.name} {r.status_code} {r.headers.get('x-request-id')}: {r.text[:300]}")
        status = r.status_code if r is not None else "no response"
        retry_after = r.headers.get("retry-after") if r is not None else None
        if retry_after and retry_after.isdigit():
            time.sleep(int(retry_after) + random.random())
        else:
            time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"{api.name}: gave up after {max_tries} tries (last: {status})")


def query(api, contexts, questions, split, workers=8, per_minute=None, cache_dir=CACHE_DIR):
    """contexts: {record id: dict}. Returns a DataFrame of P(yes), one column per question.

    Responses are appended to data/api_responses/<model>/<split>.jsonl as they arrive,
    so an interrupted run resumes where it stopped, and a complete cache needs no key.
    Calls are spaced to stay under per_minute (Grayson allows 300 requests a minute); it defaults
    to $API_PER_MINUTE, or 280, so jobs sharing a key can split the limit.
    """
    path = Path(cache_dir) / api.name / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if set(rec["probs"]) == set(questions):
                done[rec["id"]] = rec["probs"]
    todo = [i for i in contexts if i not in done]
    if todo and not api.available():
        raise MissingKey(f"{len(todo)} {api.name} responses missing from {path} and ${api.key_env} is not set")

    lock = threading.Lock()
    session = requests.Session()
    throttle = Throttle(per_minute or int(os.environ.get("API_PER_MINUTE", 280)))

    def work(tid):
        resp = _call(api, contexts[tid], questions, session, throttle)
        rec = {
            "id": int(tid),
            "model": resp.get("model"),
            "probs": api.parse(resp),
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return tid, rec["probs"]

    if todo:
        ex = ThreadPoolExecutor(workers)
        futures = [ex.submit(work, t) for t in todo]
        try:
            for n, fut in enumerate(as_completed(futures), 1):
                tid, probs = fut.result()
                done[tid] = probs
                if n % 250 == 0 or n == len(todo):
                    print(f"{api.name}/{split}: {n}/{len(todo)}")
        finally:
            # On the first failure, drop queued records instead of waiting for all of them.
            ex.shutdown(wait=True, cancel_futures=True)

    return pd.DataFrame.from_dict({i: done[i] for i in contexts}, orient="index")[list(questions)]
