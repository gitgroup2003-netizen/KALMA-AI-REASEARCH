#!/usr/bin/env python3
"""
server.py — Research Agent API

A small FastAPI server that wraps the plan -> execute -> synthesize pipeline
(same logic as research_agent.py) and exposes it as an HTTP API, so the web
app can call it directly instead of using canned demo data.

Run locally:
    pip install -r requirements-server.txt
    export OPENAI_API_KEY="sk-..."
    export SERPAPI_API_KEY="..."
    uvicorn server:app --reload --port 8000

Then point the frontend's API_BASE (see index.html) at:
    http://localhost:8000

Endpoints:
    GET  /api/health            -> {"status": "ok"}
    POST /api/research          -> runs the full pipeline, returns report + findings

Request body for /api/research:
    {
      "topic": "solid-state EV batteries",
      "mode": "deep" | "quick" | "compare" | "lit",
      "level": "Beginner" | "Practitioner" | "Expert",
      "angle": "General understanding" | "Making a decision" | "Writing or citing it",
      "detail": "optional free-text focus"
    }
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Research Agent API")

# CORS: wide open by default so you can call this from any frontend origin
# while you're testing. Narrow this to your actual frontend's domain before
# putting this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = os.environ.get("RESEARCH_MODEL", "gpt-4o-mini")
MAX_QUERIES = {"quick": 3, "deep": 5, "compare": 5, "lit": 6}


class ResearchRequest(BaseModel):
    topic: str
    mode: str = "deep"
    level: str = "Practitioner"
    angle: str = "General understanding"
    detail: str | None = None


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "Server is missing OPENAI_API_KEY.")
    return OpenAI(api_key=key)


def get_serpapi_key() -> str:
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        raise HTTPException(500, "Server is missing SERPAPI_API_KEY.")
    return key


PLAN_PROMPT = """You are a research planning assistant. Given a topic, produce a \
short list of distinct, high-value web search queries that together give thorough \
coverage. Keep each query short (2-6 words), like something a person would type \
into a search engine.

Respond ONLY with JSON: {"queries": ["query one", "query two", ...]}"""


def plan_queries(client: OpenAI, topic: str, n: int) -> list[str]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nProduce at most {n} queries."},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        queries = [q.strip() for q in json.loads(raw).get("queries", []) if q.strip()]
    except json.JSONDecodeError:
        queries = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
    return (queries or [topic])[:n]


def search_one(query: str, api_key: str, num: int = 5) -> list[dict]:
    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": query, "api_key": api_key, "num": num},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        print(f"[warn] search failed for {query!r}: {e}", file=sys.stderr)
        return []
    return [
        {"query": query, "title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")}
        for it in data.get("organic_results", [])[:num]
    ]


def execute_searches(queries: list[str], api_key: str) -> list[dict]:
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(queries) or 1)) as pool:
        futs = {pool.submit(search_one, q, api_key): q for q in queries}
        for f in concurrent.futures.as_completed(futs):
            results.extend(f.result())
    return results


SYNTH_PROMPT = """You are a research analyst. Using ONLY the given search results, \
write a concise report (250-400 words) in markdown that synthesizes the findings. \
Tailor tone and depth to the reader's level: Beginner = plain language, explain terms; \
Practitioner = balance context with practical detail; Expert = skip basics, go deep. \
Tailor emphasis to their purpose: "Making a decision" = surface trade-offs; \
"Writing or citing it" = make claims easy to cite; "General understanding" = give the \
shape of the topic. If the person gave a specific focus, address it directly. \
Cite sources inline by domain, e.g. "(example.com)". Do not invent facts beyond the \
provided results.

After the report, output a line that says exactly: ---FINDINGS_JSON---
Then a JSON object: {"topic": "...", "findings": [{"claim": "...", "source": "...", \
"category": "..."}]} with 5-8 findings tied to real sources you were given."""


def synthesize(client: OpenAI, topic: str, results: list[dict], level: str, angle: str, detail: str | None) -> tuple[str, dict]:
    blob = "\n".join(f"- [{r['query']}] {r['title']} ({r['link']}): {r['snippet']}" for r in results)
    user_context = f"Topic: {topic}\nReader level: {level}\nPurpose: {angle}\n"
    if detail:
        user_context += f"Specific focus requested: {detail}\n"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTH_PROMPT},
            {"role": "user", "content": f"{user_context}\nSearch results:\n{blob}"},
        ],
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip()
    if "---FINDINGS_JSON---" in raw:
        report_md, json_part = raw.split("---FINDINGS_JSON---", 1)
    else:
        report_md, json_part = raw, "{}"
    json_part = json_part.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        findings = json.loads(json_part)
    except json.JSONDecodeError:
        findings = {"topic": topic, "findings": []}
    return report_md.strip(), findings


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/research")
def research(req: ResearchRequest):
    if not req.topic.strip():
        raise HTTPException(400, "topic is required")

    client = get_client()
    serp_key = get_serpapi_key()
    n = MAX_QUERIES.get(req.mode, 5)

    queries = plan_queries(client, req.topic, n)
    results = execute_searches(queries, serp_key)
    if not results:
        raise HTTPException(502, "No search results came back — check SERPAPI_API_KEY / quota.")

    report_md, findings = synthesize(client, req.topic, results, req.level, req.angle, req.detail)

    sources = [{"domain": r["link"].split("/")[2] if "://" in r["link"] else r["link"], "title": r["title"], "query": r["query"]} for r in results]

    return {
        "topic": req.topic,
        "mode": req.mode,
        "level": req.level,
        "angle": req.angle,
        "queries": queries,
        "sources": sources[:12],
        "report_markdown": report_md,
        "findings": findings,
    }
