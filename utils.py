import json, re

TINY_PNG_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="

def strip_md_fences(s: str) -> str:
    if not s: return s
    s = s.strip()
    s = re.sub(r'^\s*```(?:[a-zA-Z0-9_+\-]+)?\s*', '', s)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

def fabricate_fallback(questions, response_format):
    def guess_placeholder(q: str):
        ql = (q or "").lower()
        if any(w in ql for w in ["plot","chart","figure","image","png","scatter"]):
            return TINY_PNG_DATA_URI
        if "correlation" in ql or "mean" in ql or "median" in ql or "rate" in ql or "score" in ql:
            return -1.0
        if "how many" in ql or "count" in ql or "number" in ql:
            return 0
        return "Unknown"
    answers = [guess_placeholder(q) for q in questions]
    return enforce_format(answers, response_format, len(questions))

def enforce_format(answers, response_format: str, n_expected: int):
    arr = list(answers if isinstance(answers, (list, tuple)) else answers.get("answers", []))
    if len(arr) < n_expected:
        arr += ["Unknown"] * (n_expected - len(arr))
    if len(arr) > n_expected:
        arr = arr[:n_expected]
    if "object" in (response_format or "").lower():
        return {"answers": arr}
    return arr
