import importlib.util, pathlib, json

# Load strict_enforce from app.py
spec = importlib.util.spec_from_file_location("app_mod", str(pathlib.Path(__file__).resolve().parents[1] / "app.py"))
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)

def test_strict_enforce_shapes():
    qs = ["how many?", "give correlation", "draw a plot", "free text"]
    cand = ["ten", "NaN", "not-a-uri", None]
    out = app_mod.strict_enforce(cand, qs, "JSON array")
    assert isinstance(out, list) and len(out)==4
    assert isinstance(out[0], int)
    assert isinstance(out[1], float)
    assert isinstance(out[2], str) and out[2].startswith("data:image/png;base64,")
    assert isinstance(out[3], str)

def test_strict_enforce_object():
    qs = ["plot please"]
    cand = {"answers": ["garbage"]}
    out = app_mod.strict_enforce(cand, qs, "JSON object")
    assert isinstance(out, dict) and isinstance(out["answers"], list) and len(out["answers"])==1
    assert out["answers"][0].startswith("data:image/png;base64,")
