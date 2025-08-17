import os, logging, requests, json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

log = logging.getLogger("llm")

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")

@dataclass
class LLMRoute:
    provider: str  # "openai" | "gemini"
    model: str
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        if self.provider == "openai":
            return bool(self.openai_api_key and self.model)
        if self.provider == "gemini":
            return bool(self.gemini_api_key and self.model)
        return False

def _headers_openai(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

def _responses_text(data: dict) -> str:
    if isinstance(data, dict) and data.get("output_text"):
        return data["output_text"]
    try:
        chunks = []
        for item in data.get("output", []):
            for part in item.get("content", []):
                t = part.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        if chunks:
            return "".join(chunks)
    except Exception:
        pass
    for k in ("content", "message", "text"):
        v = data.get(k)
        if isinstance(v, str):
            return v
    return ""

def _call_openai(messages: List[Dict[str, str]], model: str, api_key: str, timeout: int) -> str:
    use_responses = model.startswith("o") and not model.startswith("gpt-4o")
    if use_responses:
        url = f"{OPENAI_BASE}/v1/responses"
        payload = {
            "model": model,
            "input": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": 0,
            "response_format": {"type": "text"},
        }
        r = requests.post(url, headers=_headers_openai(api_key), json=payload, timeout=timeout)
        r.raise_for_status()
        return _responses_text(r.json()) or ""
    else:
        url = f"{OPENAI_BASE}/v1/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0}
        r = requests.post(url, headers=_headers_openai(api_key), json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

def _call_gemini(messages: List[Dict[str, str]], model: str, api_key: str, timeout: int) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        log.error("Gemini SDK not available. Add google-generativeai to requirements.txt")
        raise
    genai.configure(api_key=api_key)
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    prompt = (sys + "\n\n" + user).strip() if sys else user
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(prompt, request_options={"timeout": float(timeout)})
    return getattr(resp, "text", "") or ""

def chat_complete(messages: List[Dict[str, str]], route: LLMRoute, timeout: int = 30) -> str:
    try:
        if route.provider == "openai":
            return _call_openai(messages, route.model, route.openai_api_key, timeout)
        elif route.provider == "gemini":
            return _call_gemini(messages, route.model, route.gemini_api_key, timeout)
        else:
            raise ValueError(f"Unknown provider: {route.provider}")
    except requests.HTTPError as e:
        try:
            err = e.response.json()
        except Exception:
            err = {"error": {"message": e.response.text if e.response is not None else str(e)}}
        log.error("chat_complete failed: %s", err)
        raise
    except Exception as e:
        log.error("chat_complete unexpected error: %s", e)
        raise
