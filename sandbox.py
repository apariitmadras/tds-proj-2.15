import tempfile, subprocess, sys, os

def run_user_code(code: str, timeout: int = 60):
    code = code.lstrip("\ufeff").strip() + "\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "job.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                text=True,
                env={**os.environ},
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            return "", f"Timeout after {timeout}s", 124
