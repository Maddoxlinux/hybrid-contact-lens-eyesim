"""
Local preview server — run the whole Vercel app on your machine.

    cd vercel-app
    python devserver.py

then open  http://localhost:8000  in your browser.

Serves index.html and the static files, and routes /api/simulate to the same
Python function used on Vercel. Needs only numpy (pip install numpy).
"""
import os, sys, json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "api"))
from _compute import simulate

PORT = int(os.environ.get("PORT", 8000))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/simulate"):
            try:
                return self._json(200, simulate({}))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/simulate"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                params = json.loads(self.rfile.read(n) or b"{}")
                return self._json(200, simulate(params))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        self.send_error(404)


if __name__ == "__main__":
    print(f"Preview running at  http://localhost:{PORT}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
