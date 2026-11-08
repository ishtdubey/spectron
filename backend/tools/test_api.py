"""
End-to-end test for the research backend (server.py). Standard library only.

Start the backend first, then run this script:

    # no Groq key needed (canned data):
    MOCK=1 uvicorn server:app --port 8000
    python tools/test_api.py

    # real pipeline (needs GROQ_API_KEY in backend/.env):
    uvicorn server:app --port 8000
    python tools/test_api.py --real

Options:
    --url http://127.0.0.1:8000   backend address
    --real                        expect real-pipeline behaviour (skips mock-only checks)

Exit code is 0 if every check passes, 1 otherwise.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

STAGE_ORDER = ["classify", "plan", "collect", "critique", "synthesize", "epistemic"]
RESULT_KEYS = [
    "query_type", "perspectives", "evidence", "validated_evidence_ids",
    "conflicts", "weak_evidence", "answer", "limitations", "epistemic",
]
EPISTEMIC_KEYS = ["known_facts", "uncertainties", "disputed_claims", "unresolved_questions"]

SIMPLE_Q = "Capital of Australia?"
COMPLEX_Q = ("What are the trade-offs of moving a core financial system "
             "from microservices to a modular monolith?")

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  -> {detail}" if detail else ""))


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.load(r)


def post_stream(url, payload, timeout=180):
    """POST and return (http_status, [events]). Parses newline-delimited JSON."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            events = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
            return r.status, events
    except urllib.error.HTTPError as e:
        return e.code, []


def validate_result(res):
    """Return a list of problems with a result event (empty list = valid)."""
    problems = [f"missing '{k}'" for k in RESULT_KEYS if k not in res]
    if problems:
        return problems
    if res["query_type"] not in ("SIMPLE", "COMPLEX"):
        problems.append(f"bad query_type {res['query_type']!r}")
    for k in ("perspectives", "evidence", "validated_evidence_ids", "conflicts",
              "weak_evidence", "limitations"):
        if not isinstance(res[k], list):
            problems.append(f"'{k}' is not a list")
    if not isinstance(res["answer"], str) or not res["answer"].strip():
        problems.append("'answer' is empty")
    for k in EPISTEMIC_KEYS:
        if not isinstance(res["epistemic"].get(k), list):
            problems.append(f"epistemic.{k} is not a list")
    ids = {e.get("id") for e in res["evidence"]}
    stray = [i for i in res["validated_evidence_ids"] if i not in ids]
    if stray:
        problems.append(f"validated ids not in evidence: {stray}")
    for e in res["evidence"]:
        for k in ("id", "source", "content"):
            if k not in e:
                problems.append(f"evidence item missing '{k}'")
                break
    return problems


def run_query(base, query, label):
    print(f"\n[{label}] {query}")
    status, events = post_stream(f"{base}/api/query", {"query": query})
    check("HTTP 200", status == 200, f"got {status}")
    types = [e.get("type") for e in events]
    errors = [e for e in events if e.get("type") == "error"]
    check("no error event", not errors, errors[0]["message"] if errors else "")
    check("ends with exactly one result", types.count("result") == 1 and types[-1] == "result",
          f"types={types}")
    stages = [e["stage"] for e in events if e.get("type") == "stage"]
    check("stages are known and in pipeline order",
          all(s in STAGE_ORDER for s in stages)
          and stages == sorted(stages, key=STAGE_ORDER.index),
          f"stages={stages}")
    check("classify comes first", bool(stages) and stages[0] == "classify", f"stages={stages}")
    return events, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"Testing {base}  ({'real pipeline' if args.real else 'mock backend'})")

    # 1. health
    print("\n[health]")
    try:
        status, body = get_json(f"{base}/api/health")
    except Exception as e:
        print(f"  FAIL  cannot reach backend: {e}\n  Is uvicorn running on this port?")
        sys.exit(1)
    check("GET /api/health is 200 and ok", status == 200 and body.get("ok") is True, str(body))
    if not args.real:
        check("backend is in MOCK mode", body.get("mock") is True,
              "start it with MOCK=1, or pass --real")

    # 2. simple query -> direct path
    events, stages = run_query(base, SIMPLE_Q, "simple query")
    res = next((e for e in events if e.get("type") == "result"), None)
    if res:
        problems = validate_result(res)
        check("result has the full expected shape", not problems, "; ".join(problems))
        if not args.real:
            check("classified SIMPLE", res["query_type"] == "SIMPLE", res["query_type"])
        if res["query_type"] == "SIMPLE":
            check("no perspective stage on direct path", "plan" not in stages, str(stages))
            check("no perspectives listed", res["perspectives"] == [], str(res["perspectives"]))

    # 3. complex query -> multi-perspective path
    events, stages = run_query(base, COMPLEX_Q, "complex query")
    res = next((e for e in events if e.get("type") == "result"), None)
    if res:
        problems = validate_result(res)
        check("result has the full expected shape", not problems, "; ".join(problems))
        if not args.real:
            check("classified COMPLEX", res["query_type"] == "COMPLEX", res["query_type"])
        if res["query_type"] == "COMPLEX":
            check("perspective stage ran", "plan" in stages, str(stages))
            check("at least 2 perspectives", len(res["perspectives"]) >= 2,
                  str(res["perspectives"]))
        check("limitations are reported", len(res["limitations"]) > 0,
              "empty limitations list (spec requires a self-report on every answer)")

    # 4. input validation
    print("\n[input validation]")
    status, _ = post_stream(f"{base}/api/query", {"query": ""})
    check("empty query rejected (422)", status == 422, f"got {status}")
    status, _ = post_stream(f"{base}/api/query", {})
    check("missing query rejected (422)", status == 422, f"got {status}")
    status, _ = post_stream(f"{base}/api/query", {"query": "x" * 2001})
    check("over-long query rejected (422)", status == 422, f"got {status}")

    # 5. the UI's "render only what is real" rule: citations must resolve
    print("\n[citations]")
    if res:
        import re
        cited = set(re.findall(r"\[([^\]\s]+)\]", res["answer"]))
        known = {e["id"] for e in res["evidence"]}
        unknown = sorted(cited - known)
        check("every [doc_id] in the answer exists in the evidence", not unknown,
              f"unknown citations (UI will show them red): {unknown}")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
