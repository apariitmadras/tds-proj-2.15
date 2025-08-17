from utils import enforce_format, fabricate_fallback, TINY_PNG_DATA_URI

def test_enforce_array():
    out = enforce_format([1,2], "JSON array", 3)
    assert isinstance(out, list) and len(out) == 3

def test_enforce_object():
    out = enforce_format([1,2,3], "JSON object", 3)
    assert isinstance(out, dict) and len(out["answers"]) == 3

def test_fallbacks():
    qs = ["how many items?", "give correlation", "draw a plot"]
    out = fabricate_fallback(qs, "JSON array")
    assert out[0] == 0
    assert out[1] == -1.0
    assert isinstance(out[2], str) and out[2].startswith("data:image/png;base64,")
